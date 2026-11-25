"""Run the Hodge branch-response gate for the unactivated relative field."""

from pathlib import Path

import cube_incidence_hodge_branch_response as response


HERE = Path(__file__).resolve().parent
response.HODGE_INDEX = 2
response.CACHE = HERE/"cube_incidence_hodge_unactivated_branch_response_matrices.npz"
response.OUTPUT = HERE/"cube_incidence_hodge_unactivated_branch_response.json"


if __name__ == "__main__":
    response.efficient_main()
