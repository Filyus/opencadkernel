//! A box whose corner stands in a cone: the two plane–cone meetings each end
//! at the corner, inside the cone's face, and only joined do they divide it.

use opencadkernel::brep::{self, make};

#[test]
fn a_box_corner_in_a_cone_combines_every_way() {
    let cone = make::cone([-651.872, 190.592, 0.0], 13.395, 38.717).unwrap();
    let bx = make::cuboid([-644.233, 169.547, 0.0], [21.424, 17.697, 23.373]).unwrap();
    let tolerance = brep::operation_tolerance(&[&cone, &bx]);
    let operations =
        [brep::Operation::Union, brep::Operation::Difference, brep::Operation::Intersection];
    for how in operations {
        let result = brep::combine(cone.clone(), bx.clone(), how, tolerance)
            .unwrap_or_else(|snag| panic!("{how:?}: {snag:?}"));
        assert!(result.validate().is_empty(), "{how:?}: {:?}", result.validate());
        let mesh = brep::mesh::tessellate(
            &result,
            brep::mesh::TessellationTolerance::new(0.1, 1e-3),
        );
        assert!(mesh.missing_faces.is_empty(), "{how:?}: a face would not mesh");
    }
}
