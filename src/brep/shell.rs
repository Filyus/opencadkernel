//! Hollowing exact analytic solids.
//!
//! A shell is the material between an exterior and an offset cavity.  A
//! removed face opens that cavity through the corresponding side instead of
//! leaving a cap.  Construction stays in the kernel so every application and
//! file-format consumer receives the same topology and refusal rules.

use super::{Body, FaceKey, Operation, Placement, Snag, Surface};
use crate::space::Vec3;
use std::collections::{HashMap, HashSet};

/// Why a solid could not be hollowed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShellError {
    InvalidDistance,
    UnsupportedSolid,
    UnknownFace,
    NoMaterial,
    Kernel(Snag),
}

impl From<Snag> for ShellError {
    fn from(value: Snag) -> Self {
        Self::Kernel(value)
    }
}

#[derive(Clone, Copy)]
struct BoxFace {
    axis: usize,
    maximum: bool,
}

struct BoxShape {
    origin: [f64; 3],
    axes: [[f64; 3]; 3],
    size: [f64; 3],
    faces: HashMap<FaceKey, BoxFace>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CylinderFace {
    Side,
    Bottom,
    Top,
}

struct CylinderShape {
    origin: [f64; 3],
    axes: [[f64; 3]; 3],
    radius: f64,
    height: f64,
    faces: HashMap<FaceKey, CylinderFace>,
}

struct SphereShape {
    centre: [f64; 3],
    radius: f64,
    face: FaceKey,
}

/// Hollow `body` by `distance`, opening the cavity through `removed_faces`.
///
/// Positive distances keep the exterior fixed and offset the cavity inward.
/// Negative distances keep the original body as the cavity and grow a new
/// exterior.  Exact rectangular boxes, circular cylinders and spheres are
/// supported; other surface sets are refused without changing the input.
pub fn shell(
    body: &Body,
    removed_faces: &[FaceKey],
    distance: f64,
) -> Result<Body, ShellError> {
    if !distance.is_finite() || distance == 0.0 {
        return Err(ShellError::InvalidDistance);
    }
    let removed = removed_faces.iter().copied().collect::<HashSet<_>>();
    if removed.len() != removed_faces.len()
        || removed.iter().any(|face| !body.faces.contains(*face))
    {
        return Err(ShellError::UnknownFace);
    }
    if body.roots.len() != 1 || body.lumps.len() != 1 || body.shells.len() != 1 {
        return Err(ShellError::UnsupportedSolid);
    }

    let result = if let Some(shape) = recognize_box(body) {
        shell_box(body, &shape, &removed, distance)?
    } else if let Some(shape) = recognize_cylinder(body) {
        shell_cylinder(body, &shape, &removed, distance)?
    } else if let Some(shape) = recognize_sphere(body) {
        shell_sphere(body, &shape, &removed, distance)?
    } else if let Some(result) = shell_prismatic(body, &removed, distance) {
        result?
    } else {
        shell_polyhedral(body, &removed, distance)?
    };
    if result.face_keys().next().is_none() || result.roots.is_empty() {
        return Err(ShellError::NoMaterial);
    }
    Ok(result)
}

/// Resolve a surface pick to a face supported by [`shell`].
pub fn shell_face_at_point(body: &Body, point: [f64; 3]) -> Option<FaceKey> {
    if point.iter().any(|value| !value.is_finite())
        || (recognize_box(body).is_none()
            && recognize_cylinder(body).is_none()
            && recognize_sphere(body).is_none())
    {
        return None;
    }
    body.face_keys()
        .filter_map(|key| {
            let face = body.faces.get(key)?;
            let surface = body.surfaces.get(face.surface)?;
            Some((key, surface.distance_to(point).abs()))
        })
        .min_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(key, _)| key)
}

fn shell_box(
    body: &Body,
    shape: &BoxShape,
    removed: &HashSet<FaceKey>,
    distance: f64,
) -> Result<Body, ShellError> {
    let mut open = [[false; 2]; 3];
    for face in removed {
        let Some(tag) = shape.faces.get(face) else {
            return Err(ShellError::UnknownFace);
        };
        open[tag.axis][usize::from(tag.maximum)] = true;
    }
    if removed.len() == shape.faces.len() {
        return Err(ShellError::NoMaterial);
    }
    let thickness = distance.abs();
    let span = shape.size.iter().copied().fold(0.0, f64::max);
    let extension = span + thickness * 2.0;

    let (outer_min, outer_size, cavity_min, cavity_size) = if distance > 0.0 {
        let mut cavity_min = [0.0; 3];
        let mut cavity_max = [0.0; 3];
        for axis in 0..3 {
            cavity_min[axis] = if open[axis][0] { -extension } else { thickness };
            cavity_max[axis] = if open[axis][1] {
                shape.size[axis] + extension
            } else {
                shape.size[axis] - thickness
            };
            if cavity_max[axis] <= cavity_min[axis] {
                return Err(ShellError::NoMaterial);
            }
        }
        (
            [0.0; 3],
            shape.size,
            cavity_min,
            std::array::from_fn(|axis| cavity_max[axis] - cavity_min[axis]),
        )
    } else {
        let outer_min = [-thickness; 3];
        let outer_size = std::array::from_fn(|axis| shape.size[axis] + 2.0 * thickness);
        let mut cavity_min = [0.0; 3];
        let mut cavity_max = shape.size;
        for axis in 0..3 {
            if open[axis][0] {
                cavity_min[axis] = -extension;
            }
            if open[axis][1] {
                cavity_max[axis] = shape.size[axis] + extension;
            }
        }
        (
            outer_min,
            outer_size,
            cavity_min,
            std::array::from_fn(|axis| cavity_max[axis] - cavity_min[axis]),
        )
    };

    let outer = placed_box(shape, outer_min, outer_size)?;
    let cavity = placed_box(shape, cavity_min, cavity_size)?;
    subtract(outer, cavity, body)
}

fn placed_box(shape: &BoxShape, local_origin: [f64; 3], size: [f64; 3]) -> Result<Body, ShellError> {
    let local = super::make::cuboid(local_origin, size).ok_or(ShellError::NoMaterial)?;
    super::place::transform(
        &local,
        &Placement {
            x_axis: shape.axes[0],
            y_axis: shape.axes[1],
            z_axis: shape.axes[2],
            origin: shape.origin,
        },
    )
    .ok_or(ShellError::UnsupportedSolid)
}

fn shell_cylinder(
    body: &Body,
    shape: &CylinderShape,
    removed: &HashSet<FaceKey>,
    distance: f64,
) -> Result<Body, ShellError> {
    let mut side_open = false;
    let mut bottom_open = false;
    let mut top_open = false;
    for face in removed {
        match shape.faces.get(face).copied().ok_or(ShellError::UnknownFace)? {
            CylinderFace::Side => side_open = true,
            CylinderFace::Bottom => bottom_open = true,
            CylinderFace::Top => top_open = true,
        }
    }
    if side_open && bottom_open && top_open {
        return Err(ShellError::NoMaterial);
    }
    let thickness = distance.abs();
    let extension = shape.radius.max(shape.height) + thickness * 2.0;
    let (outer_radius, outer_z, outer_height, cavity_radius, cavity_z, cavity_height) =
        if distance > 0.0 {
            let cavity_radius = if side_open {
                shape.radius + extension
            } else {
                shape.radius - thickness
            };
            let cavity_z = if bottom_open { -extension } else { thickness };
            let cavity_top = if top_open {
                shape.height + extension
            } else {
                shape.height - thickness
            };
            (
                shape.radius,
                0.0,
                shape.height,
                cavity_radius,
                cavity_z,
                cavity_top - cavity_z,
            )
        } else {
            let cavity_radius = if side_open {
                shape.radius + thickness + extension
            } else {
                shape.radius
            };
            let cavity_z = if bottom_open { -extension } else { 0.0 };
            let cavity_top = if top_open {
                shape.height + extension
            } else {
                shape.height
            };
            (
                shape.radius + thickness,
                -thickness,
                shape.height + 2.0 * thickness,
                cavity_radius,
                cavity_z,
                cavity_top - cavity_z,
            )
        };
    if outer_radius <= 0.0
        || outer_height <= 0.0
        || cavity_radius <= 0.0
        || cavity_height <= 0.0
    {
        return Err(ShellError::NoMaterial);
    }
    let outer = placed_cylinder(shape, outer_radius, outer_z, outer_height)?;
    let cavity = placed_cylinder(shape, cavity_radius, cavity_z, cavity_height)?;
    subtract(outer, cavity, body)
}

fn placed_cylinder(
    shape: &CylinderShape,
    radius: f64,
    local_z: f64,
    height: f64,
) -> Result<Body, ShellError> {
    let local = super::make::cylinder([0.0, 0.0, local_z], radius, height)
        .ok_or(ShellError::NoMaterial)?;
    super::place::transform(
        &local,
        &Placement {
            x_axis: shape.axes[0],
            y_axis: shape.axes[1],
            z_axis: shape.axes[2],
            origin: shape.origin,
        },
    )
    .ok_or(ShellError::UnsupportedSolid)
}

fn shell_sphere(
    body: &Body,
    shape: &SphereShape,
    removed: &HashSet<FaceKey>,
    distance: f64,
) -> Result<Body, ShellError> {
    if !removed.is_empty() {
        return if removed.contains(&shape.face) {
            Err(ShellError::NoMaterial)
        } else {
            Err(ShellError::UnknownFace)
        };
    }
    let thickness = distance.abs();
    let (outer_radius, cavity_radius) = if distance > 0.0 {
        (shape.radius, shape.radius - thickness)
    } else {
        (shape.radius + thickness, shape.radius)
    };
    if outer_radius <= 0.0 || cavity_radius <= 0.0 {
        return Err(ShellError::NoMaterial);
    }
    let outer = super::make::sphere(shape.centre, outer_radius).ok_or(ShellError::NoMaterial)?;
    let cavity = super::make::sphere(shape.centre, cavity_radius).ok_or(ShellError::NoMaterial)?;
    subtract(outer, cavity, body)
}

/// A solid run straight between two parallel caps — its walls planes and
/// cylinders square to them, a plate with holes, a wall bent round an arc:
/// the cavity is the cap's region offset in by the thickness, run between
/// the caps moved in, or out through a removed cap. Offset in the plane,
/// a wall narrower than the shell closes up there and a hole's wall grows
/// round it as the region allows. Not taken (`None`) for any other solid, or
/// where a wall is to be opened.
#[cfg(feature = "offset")]
fn shell_prismatic(
    body: &Body,
    removed: &HashSet<FaceKey>,
    distance: f64,
) -> Option<Result<Body, ShellError>> {
    use crate::geom2d::{Curve, Polyline};
    let outward = |face: FaceKey| -> Option<Vec3> {
        let node = body.faces.get(face)?;
        let Some(Surface::Plane(plane)) = body.surfaces.get(node.surface) else {
            return None;
        };
        Some(Vec3::from(plane.normal()?) * if node.forward { 1.0 } else { -1.0 })
    };
    // The caps: a plane, the one plane facing the other way, and walls
    // square to both.
    let caps = body.face_keys().find_map(|top| {
        let normal = outward(top)?;
        let mut bottom = None;
        for face in body.face_keys().filter(|face| *face != top) {
            match body.faces.get(face).and_then(|node| body.surfaces.get(node.surface))? {
                Surface::Plane(_) => {
                    let along = outward(face)?.dot(normal);
                    if along < -1.0 + 1e-9 && bottom.is_none() {
                        bottom = Some(face);
                    } else if along.abs() > 1e-9 {
                        return None;
                    }
                }
                Surface::Cylinder(cylinder) => {
                    if Vec3::from(cylinder.base.normal()?).dot(normal).abs() < 1.0 - 1e-9 {
                        return None;
                    }
                }
                _ => return None,
            }
        }
        Some((top, bottom?, normal))
    });
    let (top, bottom, normal) = caps?;
    if removed.iter().any(|face| *face != top && *face != bottom) {
        return None;
    }
    let profile = super::planar_face_profile(body, top)?;
    let floor = body.faces.get(bottom).and_then(|node| body.surfaces.get(node.surface));
    let Some(Surface::Plane(floor)) = floor else {
        return None;
    };
    let height = (Vec3::from(profile.plane.origin) - Vec3::from(floor.origin)).dot(normal);
    let rings = profile.loops.iter().map(|ring| ring_polyline(ring)).collect::<Option<Vec<_>>>()?;
    let tolerance = super::operation_tolerance(&[body]);
    let thickness = distance.abs();
    let region: Vec<Curve> = rings.iter().cloned().map(Curve::Polyline).collect();
    let in_region = |point: [f64; 2]| {
        crate::geom2d::contains(&region, point, crate::geom2d::Tolerance::new(tolerance))
    };
    // Each ring moved off its material, or into it.
    let moved = |ring: &Polyline, into: bool| -> Vec<Polyline> {
        let curve = Curve::Polyline(ring.clone());
        let pieces = curve.segment_count().max(1) as f64;
        let (at, ahead) = (curve.point_at(0.5 / pieces), curve.point_at(0.5 / pieces + 1e-6));
        let along = [ahead[0] - at[0], ahead[1] - at[1]];
        let length = along[0].hypot(along[1]);
        if length == 0.0 {
            return Vec::new();
        }
        let reach = thickness * 1e-3;
        let left = [at[0] - along[1] / length * reach, at[1] + along[0] / length * reach];
        let right = [at[0] + along[1] / length * reach, at[1] - along[0] / length * reach];
        let side = if in_region(left) == into { left } else { right };
        crate::geom2d::offset_polyline(ring, thickness, side)
    };
    // A region's ring run along the cap's normal between two depths below
    // the top cap.
    let run = |ring: &Polyline, from: f64, to: f64| -> Option<Body> {
        let mut plane = profile.plane;
        plane.origin = (Vec3::from(plane.origin) - normal * from).to_array();
        let along = (normal * (from - to)).to_array();
        super::extrude_region(plane, &[vec![Curve::Polyline(ring.clone())]], along)
    };
    // The region of `outer` rings less `holes` rings, run between depths.
    let solid = |outer: &[Polyline], holes: &[Polyline], from: f64, to: f64| -> Option<Body> {
        let mut pieces = outer.iter().map(|ring| run(ring, from, to));
        let mut result = pieces.next()??;
        for piece in pieces {
            result = super::combine(result, piece?, Operation::Union, tolerance).ok()?;
        }
        for hole in holes {
            result = super::combine(result, run(hole, from, to)?, Operation::Difference, tolerance)
                .ok()?;
        }
        Some(result)
    };
    let (open_top, open_bottom) = (removed.contains(&top), removed.contains(&bottom));
    let through = thickness * 2.0;
    let built = if distance > 0.0 {
        let outer = moved(&rings[0], true);
        if outer.is_empty() {
            return Some(Err(ShellError::NoMaterial));
        }
        let holes: Vec<Polyline> = rings[1..].iter().flat_map(|ring| moved(ring, true)).collect();
        let (from, to) = (
            if open_top { -through } else { thickness },
            if open_bottom { height + through } else { height - thickness },
        );
        if to <= from {
            return Some(Err(ShellError::NoMaterial));
        }
        solid(&outer, &holes, from, to).map(|cavity| (body.clone(), cavity))
    } else {
        let outer = moved(&rings[0], false);
        let holes: Vec<Polyline> = rings[1..].iter().flat_map(|ring| moved(ring, false)).collect();
        let exterior = solid(&outer, &holes, -thickness, height + thickness);
        let cavity = if removed.is_empty() {
            Some(body.clone())
        } else {
            solid(
                &rings[..1],
                &rings[1..],
                if open_top { -through - thickness } else { 0.0 },
                if open_bottom { height + through + thickness } else { height },
            )
        };
        exterior.zip(cavity)
    };
    let Some((exterior, cavity)) = built else {
        return Some(Err(ShellError::UnsupportedSolid));
    };
    Some(hollowed(exterior, cavity, body, removed.is_empty()))
}

#[cfg(not(feature = "offset"))]
fn shell_prismatic(_: &Body, _: &HashSet<FaceKey>, _: f64) -> Option<Result<Body, ShellError>> {
    None
}

/// A profile ring as one closed polyline, arcs as bulges. `None` for a
/// piece neither straight nor round, or a ring that does not close.
#[cfg(feature = "offset")]
fn ring_polyline(ring: &[crate::geom2d::Curve]) -> Option<crate::geom2d::Polyline> {
    use crate::geom2d::{Curve, PolylineVertex};
    let pieces = super::extrusion_profile_pieces(ring);
    let ends = |piece: &Curve| (piece.point_at(0.0), piece.point_at(1.0));
    let size = pieces
        .iter()
        .flat_map(|piece| [ends(piece).0, ends(piece).1])
        .fold(0.0f64, |big, p| big.max(p[0].abs()).max(p[1].abs()));
    let near = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).hypot(a[1] - b[1]) <= 1e-9 * (1.0 + size);
    let first = pieces.first()?;
    let next = pieces.get(1).unwrap_or(first);
    let (a, b) = ends(first);
    let mut at = if near(b, ends(next).0) || near(b, ends(next).1) { a } else { b };
    let start = at;
    let mut vertices = Vec::with_capacity(pieces.len());
    for piece in &pieces {
        let (from, to) = ends(piece);
        let forward = if near(from, at) {
            true
        } else if near(to, at) {
            false
        } else {
            return None;
        };
        let bulge = match piece {
            Curve::Line(_) => 0.0,
            Curve::Arc(arc) => {
                let turn = (arc.sweep() / 4.0).tan();
                if forward { turn } else { -turn }
            }
            _ => return None,
        };
        vertices.push(PolylineVertex { position: at, bulge });
        at = if forward { to } else { from };
    }
    near(at, start).then_some(crate::geom2d::Polyline { vertices, closed: true })
}

/// The exterior less the cavity, a closed one checked to take exactly its
/// own volume out.
fn hollowed(
    exterior: Body,
    cavity: Body,
    reference: &Body,
    closed: bool,
) -> Result<Body, ShellError> {
    let hollow = subtract(exterior.clone(), cavity.clone(), reference)?;
    if closed {
        let volume = |body: &Body| super::mass_properties(body).map(|mass| mass.volume);
        let (whole, inner, left) = (
            volume(&exterior).ok_or(ShellError::NoMaterial)?,
            volume(&cavity).ok_or(ShellError::NoMaterial)?,
            volume(&hollow).ok_or(ShellError::NoMaterial)?,
        );
        if (whole - inner - left).abs() > 1e-6 * whole.abs() {
            return Err(ShellError::NoMaterial);
        }
    }
    Ok(hollow)
}

/// Any solid bounded by planes alone: the cavity is the solid with every
/// face moved in by the thickness — each corner where its faces' moved
/// planes meet — and every removed face moved out instead, so the cavity
/// runs out through it. Grown outward, the exterior is the solid moved out
/// and the cavity the solid itself, run out through the removed faces.
/// Refused where a moved face would turn over, cross itself or lose a
/// corner — the thickness eating a narrower part of the solid.
fn shell_polyhedral(
    body: &Body,
    removed: &HashSet<FaceKey>,
    distance: f64,
) -> Result<Body, ShellError> {
    let planar = body.faces.iter().all(|(_, face)| {
        matches!(body.surfaces.get(face.surface), Some(Surface::Plane(_)))
    });
    if !planar {
        return Err(ShellError::UnsupportedSolid);
    }
    if removed.len() == body.faces.len() {
        return Err(ShellError::NoMaterial);
    }
    let thickness = distance.abs();
    let through = |face: FaceKey| if removed.contains(&face) { thickness * 2.0 } else { 0.0 };
    let (outer, cavity) = if distance > 0.0 {
        let cavity = offset_planes(body, &|face| {
            if removed.contains(&face) { through(face) } else { -thickness }
        })
        .ok_or(ShellError::NoMaterial)?;
        // Every corner of the cavity not run out through an open face lies
        // inside the solid; one outside is a wall thinner than the shell.
        let tolerance = super::operation_tolerance(&[body]);
        let faces_at = corner_faces(&cavity);
        let inside = cavity.vertices.iter().all(|(key, vertex)| {
            let open = faces_at
                .get(&key)
                .is_some_and(|faces| faces.iter().any(|face| removed.contains(face)));
            open || super::contains_point(body, vertex.point, tolerance)
                == super::Containment::Inside
        });
        if !inside {
            return Err(ShellError::NoMaterial);
        }
        (body.clone(), cavity)
    } else {
        let outer = offset_planes(body, &|_| thickness).ok_or(ShellError::NoMaterial)?;
        let cavity = offset_planes(body, &through).ok_or(ShellError::NoMaterial)?;
        (outer, cavity)
    };
    hollowed(outer, cavity, body, removed.is_empty())
}

/// `body`, every face moved along its outward normal by `shift` of it and
/// every corner put where its faces' moved planes meet, keeping the
/// topology. `None` where three planes do not settle a corner, or a face
/// turns over or crosses itself on the way.
fn offset_planes(body: &Body, shift: &dyn Fn(FaceKey) -> f64) -> Option<Body> {
    let tolerance = super::operation_tolerance(&[body]);
    let mut result = body.clone();
    let mut planes: HashMap<FaceKey, (Vec3, f64)> = HashMap::new();
    for (key, face) in body.faces.iter() {
        let Some(Surface::Plane(plane)) = body.surfaces.get(face.surface) else {
            return None;
        };
        let normal = Vec3::from(plane.normal()?) * if face.forward { 1.0 } else { -1.0 };
        let moved = Vec3::from(plane.origin) + normal * shift(key);
        planes.insert(key, (normal, normal.dot(moved)));
        let mut surface = *plane;
        surface.origin = moved.to_array();
        let slot = result.surfaces.insert(Surface::Plane(surface));
        result.faces.get_mut(key)?.surface = slot;
    }
    // Each corner: the point nearest every moved plane round it, which must
    // lie on all of them.
    let faces_at = corner_faces(body);
    for (key, _) in body.vertices.iter() {
        let around: Vec<(Vec3, f64)> = faces_at
            .get(&key)
            .into_iter()
            .flatten()
            .filter_map(|face| planes.get(face).copied())
            .collect();
        if around.len() < 3 {
            return None;
        }
        let mut normal = [[0.0; 3]; 3];
        let mut right = [0.0; 3];
        for (n, c) in &around {
            let n = n.to_array();
            for row in 0..3 {
                for column in 0..3 {
                    normal[row][column] += n[row] * n[column];
                }
                right[row] += n[row] * c;
            }
        }
        let point = Vec3::from(solve3(normal, right)?);
        let scale = 1.0 + point.length();
        if around.iter().any(|(n, c)| (n.dot(point) - c).abs() > tolerance.max(1e-12 * scale)) {
            return None;
        }
        result.vertices.get_mut(key)?.point = point.to_array();
    }
    // Straight edges between the new corners.
    for (key, edge) in body.edges.iter() {
        let from = Vec3::from(result.vertices.get(edge.start)?.point);
        let to = Vec3::from(result.vertices.get(edge.end)?.point);
        let along = to - from;
        if along.length() <= tolerance {
            return None;
        }
        let curve = result.curves.insert(super::Curve3::Line(super::geometry::Line3 {
            origin: from.to_array(),
            direction: along.to_array(),
        }));
        let node = result.edges.get_mut(key)?;
        node.curve = curve;
        node.start_parameter = 0.0;
        node.end_parameter = 1.0;
        for coedge in node.coedges.clone() {
            result.coedges.get_mut(coedge)?.pcurve = None;
        }
    }
    // Each face the same way round as before, and simple.
    for (key, _) in body.faces.iter() {
        let (before, _) = super::split::planar_measure(body, key)?;
        let (after, _) = super::split::planar_measure(&result, key)?;
        if before * after <= 0.0 || after.abs() <= tolerance * tolerance {
            return None;
        }
        if folds(&result, key, tolerance)? {
            return None;
        }
    }
    result.validate().is_empty().then_some(result)
}

/// The faces meeting at each corner.
fn corner_faces(body: &Body) -> HashMap<super::VertexKey, HashSet<FaceKey>> {
    let mut faces: HashMap<super::VertexKey, HashSet<FaceKey>> = HashMap::new();
    for face in body.face_keys() {
        for coedge in body.face_coedges(face) {
            if let Some((from, to)) = body.coedge_vertices(coedge) {
                faces.entry(from).or_default().insert(face);
                faces.entry(to).or_default().insert(face);
            }
        }
    }
    faces
}

/// Whether two straight edges of a face cross anywhere but at a corner they
/// share.
fn folds(body: &Body, face: FaceKey, tolerance: f64) -> Option<bool> {
    let node = body.faces.get(face)?;
    let Some(Surface::Plane(plane)) = body.surfaces.get(node.surface) else {
        return None;
    };
    let mut segments: Vec<([f64; 2], [f64; 2])> = Vec::new();
    for coedge in body.face_coedges(face) {
        let edge = body.edges.get(body.coedges.get(coedge)?.edge)?;
        let from = plane.project(body.vertices.get(edge.start)?.point)?;
        let to = plane.project(body.vertices.get(edge.end)?.point)?;
        segments.push((from, to));
    }
    let near = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).hypot(a[1] - b[1]) <= tolerance;
    let cross = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };
    for (index, (a, b)) in segments.iter().enumerate() {
        for (c, d) in &segments[index + 1..] {
            if near(*a, *c) || near(*a, *d) || near(*b, *c) || near(*b, *d) {
                continue;
            }
            let (d1, d2) = (cross(*a, *b, *c), cross(*a, *b, *d));
            let (d3, d4) = (cross(*c, *d, *a), cross(*c, *d, *b));
            if d1 * d2 < 0.0 && d3 * d4 < 0.0 {
                return Some(true);
            }
        }
    }
    Some(false)
}

/// The solution of a 3×3 system by Cramer's rule, `None` when singular.
fn solve3(m: [[f64; 3]; 3], r: [f64; 3]) -> Option<[f64; 3]> {
    let det = |m: [[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let whole = det(m);
    let scale = m.iter().flatten().fold(0.0f64, |big, value| big.max(value.abs()));
    if whole.abs() <= 1e-12 * scale.powi(3) {
        return None;
    }
    Some(std::array::from_fn(|column| {
        let mut swapped = m;
        for row in 0..3 {
            swapped[row][column] = r[row];
        }
        det(swapped) / whole
    }))
}

fn subtract(outer: Body, cavity: Body, reference: &Body) -> Result<Body, ShellError> {
    let tolerance = super::operation_tolerance(&[reference, &outer, &cavity]);
    super::combine(outer, cavity, Operation::Difference, tolerance).map_err(ShellError::Kernel)
}

fn recognize_box(body: &Body) -> Option<BoxShape> {
    if body.faces.len() != 6 || body.edges.len() != 12 || body.vertices.len() != 8 {
        return None;
    }
    if body.face_keys().any(|key| {
        body.faces
            .get(key)
            .and_then(|face| body.surfaces.get(face.surface))
            .is_none_or(|surface| !matches!(surface, Surface::Plane(_)))
    }) {
        return None;
    }
    let (corner_key, corner) = body.vertices.iter().next()?;
    let origin = corner.point;
    let mut spans = Vec::new();
    for (_, edge) in body.edges.iter() {
        let other = if edge.start == corner_key {
            Some(edge.end)
        } else if edge.end == corner_key {
            Some(edge.start)
        } else {
            None
        };
        let Some(other) = other else { continue };
        let delta = Vec3::from(body.vertices.get(other)?.point) - Vec3::from(origin);
        spans.push((delta.normalize()?.to_array(), delta.length()));
    }
    if spans.len() != 3 {
        return None;
    }
    let axes = [spans[0].0, spans[1].0, spans[2].0];
    let size = [spans[0].1, spans[1].1, spans[2].1];
    let scale = size.iter().copied().fold(1.0, f64::max);
    let tolerance = scale * 1e-8;
    for one in 0..3 {
        for other in one + 1..3 {
            if Vec3::from(axes[one]).dot(Vec3::from(axes[other])).abs() > 1e-8 {
                return None;
            }
        }
    }
    for (_, vertex) in body.vertices.iter() {
        let offset = Vec3::from(vertex.point) - Vec3::from(origin);
        for axis in 0..3 {
            let coordinate = offset.dot(Vec3::from(axes[axis]));
            if coordinate.abs() > tolerance && (coordinate - size[axis]).abs() > tolerance {
                return None;
            }
        }
    }
    let mut faces = HashMap::new();
    for key in body.face_keys() {
        let face = body.faces.get(key)?;
        let Surface::Plane(plane) = body.surfaces.get(face.surface)? else {
            return None;
        };
        let mut normal = Vec3::from(plane.normal()?);
        if !face.forward {
            normal = -normal;
        }
        let (axis, projection) = (0..3)
            .map(|axis| (axis, normal.dot(Vec3::from(axes[axis]))))
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))?;
        if projection.abs() < 1.0 - 1e-8 {
            return None;
        }
        faces.insert(key, BoxFace { axis, maximum: projection > 0.0 });
    }
    (faces.len() == 6).then_some(BoxShape { origin, axes, size, faces })
}

fn recognize_cylinder(body: &Body) -> Option<CylinderShape> {
    if body.faces.len() != 3 {
        return None;
    }
    let mut cylinder = None;
    let mut caps = Vec::new();
    for key in body.face_keys() {
        let face = body.faces.get(key)?;
        match body.surfaces.get(face.surface)? {
            Surface::Cylinder(surface) if cylinder.is_none() => cylinder = Some((key, *surface)),
            Surface::Plane(plane) => caps.push((key, *plane)),
            _ => return None,
        }
    }
    let (side, cylinder) = cylinder?;
    if caps.len() != 2 || !cylinder.radius.is_finite() || cylinder.radius <= 0.0 {
        return None;
    }
    let axis = Vec3::from(cylinder.base.normal()?).normalize()?;
    let x = Vec3::from(cylinder.base.x_axis).normalize()?;
    let x = (x - axis * x.dot(axis)).normalize()?;
    let y = axis.cross(x).normalize()?;
    let mut cap_positions = caps
        .iter()
        .map(|(key, plane)| {
            let normal = Vec3::from(plane.normal()?);
            if normal.dot(axis).abs() < 1.0 - 1e-8 {
                return None;
            }
            Some((*key, (Vec3::from(plane.origin) - Vec3::from(cylinder.base.origin)).dot(axis)))
        })
        .collect::<Option<Vec<_>>>()?;
    cap_positions.sort_by(|a, b| a.1.total_cmp(&b.1));
    let height = cap_positions[1].1 - cap_positions[0].1;
    if !height.is_finite() || height <= 0.0 {
        return None;
    }
    let origin = (Vec3::from(cylinder.base.origin) + axis * cap_positions[0].1).to_array();
    let faces = HashMap::from([
        (side, CylinderFace::Side),
        (cap_positions[0].0, CylinderFace::Bottom),
        (cap_positions[1].0, CylinderFace::Top),
    ]);
    Some(CylinderShape {
        origin,
        axes: [x.to_array(), y.to_array(), axis.to_array()],
        radius: cylinder.radius,
        height,
        faces,
    })
}

fn recognize_sphere(body: &Body) -> Option<SphereShape> {
    if body.faces.len() != 1 {
        return None;
    }
    let face = body.face_keys().next()?;
    let node = body.faces.get(face)?;
    let Surface::Sphere(sphere) = body.surfaces.get(node.surface)? else {
        return None;
    };
    (sphere.radius.is_finite() && sphere.radius > 0.0).then_some(SphereShape {
        centre: sphere.frame.origin,
        radius: sphere.radius,
        face,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brep::{analytic_mass_properties, make};
    use std::f64::consts::PI;

    #[test]
    fn a_plate_with_two_holes_hollows_round_them() {
        // A 10 × 6 plate a unit thick with two unit holes. Moved in a
        // tenth, its region is 9.8 × 5.8 less two holes of 1.1; grown, its
        // walls moved out to meet sharp, 10.2 × 6.2 less holes of 0.9.
        let mut plate = make::cuboid([0.0, 0.0, 0.0], [10.0, 6.0, 1.0]).unwrap();
        for x in [3.0, 7.0] {
            let hole = make::cylinder([x, 3.0, -1.0], 1.0, 3.0).unwrap();
            plate = crate::brep::combine(plate, hole, Operation::Difference, 1e-9).unwrap();
        }
        let volume = |body: &Body| crate::brep::mass_properties(body).unwrap().volume;
        let solid = 60.0 - 2.0 * PI;
        let shrunk = 9.8 * 5.8 - 2.0 * PI * 1.21;
        let grown = 10.2 * 6.2 - 2.0 * PI * 0.81;
        let top = plate
            .face_keys()
            .find(|face| {
                let node = plate.faces.get(*face).unwrap();
                matches!(plate.surfaces.get(node.surface), Some(Surface::Plane(plane))
                    if (plane.origin[2] - 1.0).abs() < 1e-9)
            })
            .unwrap();
        for (removed, distance, expected) in [
            (vec![], 0.1, solid - shrunk * 0.8),
            (vec![top], 0.1, solid - shrunk * 0.9),
            (vec![], -0.1, grown * 1.2 - solid),
        ] {
            let hollow = shell(&plate, &removed, distance).unwrap();
            assert!(hollow.validate().is_empty());
            let got = volume(&hollow);
            assert!((got - expected).abs() < 1e-4 * expected, "{distance}: {got} {expected}");
        }
    }

    #[test]
    fn an_l_shaped_block_hollows_closed_open_and_outward() {
        // The L's footprint is 12, its cavity's moved in a tenth 10.44 and
        // its grown exterior's 13.64.
        let long = make::cuboid([0.0; 3], [4.0, 2.0, 1.0]).unwrap();
        let wing = make::cuboid([0.0; 3], [2.0, 4.0, 1.0]).unwrap();
        let block = crate::brep::combine(long, wing, Operation::Union, 1e-9).unwrap();
        let volume = |body: &Body| crate::brep::mass_properties(body).unwrap().volume;
        let top = block
            .face_keys()
            .find(|face| {
                let node = block.faces.get(*face).unwrap();
                matches!(block.surfaces.get(node.surface), Some(Surface::Plane(plane))
                    if (plane.origin[2] - 1.0).abs() < 1e-9)
            })
            .unwrap();
        for (removed, distance, expected) in [
            (vec![], 0.1, 12.0 - 10.44 * 0.8),
            (vec![top], 0.1, 12.0 - 10.44 * 0.9),
            (vec![], -0.1, 13.64 * 1.2 - 12.0),
        ] {
            let hollow = shell(&block, &removed, distance).unwrap();
            assert!(hollow.validate().is_empty());
            assert!((volume(&hollow) - expected).abs() < 1e-6 * expected, "{distance}");
        }
    }

    #[test]
    fn a_sphere_shell_has_the_exact_annular_volume() {
        let solid = make::sphere([3.0, -2.0, 5.0], 4.0).unwrap();
        let result = shell(&solid, &[], 1.0).unwrap();

        assert!(result.validate().is_empty());
        assert_eq!(result.shells.len(), 2);
        let properties = analytic_mass_properties(&result).unwrap();
        let expected = 4.0 * PI * (4.0_f64.powi(3) - 3.0_f64.powi(3)) / 3.0;
        assert!((properties.volume - expected).abs() < 1e-8);
        assert_eq!(properties.centroid, [3.0, -2.0, 5.0]);
    }

    #[test]
    fn removing_a_box_face_opens_a_valid_shell() {
        let solid = make::cuboid([0.0; 3], [6.0, 5.0, 4.0]).unwrap();
        let removed = shell_face_at_point(&solid, [3.0, 2.5, 4.0]).unwrap();
        let result = shell(&solid, &[removed], 0.5).unwrap();

        assert!(result.validate().is_empty());
        assert_eq!(result.roots.len(), 1);
        assert_eq!(result.lumps.len(), 1);
    }
}
