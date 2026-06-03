# O-U Conditional Probabilistic Medium Response Model

This folder contains the first-version main model for the graduate mobility
ray-tracing project.

Model scope:

- Foreign incident beams: `O != U`
  - `F_OU = F_reflection + F_absorption + F_refraction`
  - Predict `P_reflection(O,U)`, `P_absorption(O,U)`, `P_refraction(O,U)`.
- Local beams: `O = U`
  - `F_local = F_escape + F_retention`
  - Predict `P_escape(U)`, `P_retention(U)`.

This version deliberately does not use `O_code` / `U_code` as categorical
memory features. Path memory is represented through O-city attributes, U-city
attributes, and O-U relational attributes such as distance, region relation,
level gap, and city-attribute gradients.

It also does not use Dcity / `C_` destination attributes for prediction.

Run example:

```powershell
& "D:\APP\programming\Anaconda\python.exe" run_main_model.py `
  --main "..\talent_ray_2d_calibration\data\raw\ReadytoRunModel_OUD_CityLevel.csv"
```

Main outputs are written to `outputs/main_model/`.
