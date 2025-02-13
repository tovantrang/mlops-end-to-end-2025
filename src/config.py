import os

from dynaconf import Dynaconf

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    root_path=ROOT_PATH,
    settings_files=["config/settings.toml", "config/.secrets.toml"],
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
