"""Tie each reproducible notebook revision to the actual TEAM_ID in its parameter cell."""
TEAM_ID = 21
NOTEBOOK_TEAMS = {
    "9cf720e6cad44b8dec33e9a6fafc8425ce74e61d1fb37da65584b2ac9d4a73c5": TEAM_ID,
    "83e498336ed0a6081753ae49dc1dc2b93ca4066e28515a3204c799a9904876fd": 26,
}


def verified_team(report, notebook_sha):
    team = NOTEBOOK_TEAMS[notebook_sha]
    values = (report["team_id"], report["seed"], report["params"]["TEAM_ID"],
              report["train"]["train_args"]["seed"])
    if values != (team,) * len(values):
        raise ValueError(f"Notebook, training seed and report must agree: expected {team}, found {values}")
    return team
