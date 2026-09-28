"""Optional local printed-form entry point; receipts/cards retain the base.

Native words are requested only after measured rules establish a printed grid.
All refinements keep the accepted base and use the existing local libraries.
The application default is unchanged until native release verification passes.
"""
import sys
from progressive_content_mesh import solve_coordinate_staged


if __name__=='__main__':
    solve_coordinate_staged(sys.argv[1],sys.argv[2],
                            sys.argv[3] if len(sys.argv)>3 else None,
                            localization_halo=3)
