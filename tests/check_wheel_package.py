"""Check built wheel contents against the library's source and starter files."""
from configparser import ConfigParser
from pathlib import Path
import sys
from zipfile import ZipFile


def main() -> None:
    package = Path(__file__).resolve().parents[1] / "src" / "s1compiler"
    expected = {"s1compiler/" + path.relative_to(package).as_posix()
                for path in package.rglob("*") if path.is_file() and
                (path.suffix == ".py" or path.name == "py.typed" or
                 "templates" in path.relative_to(package).parts)}
    with ZipFile(sys.argv[1]) as wheel:
        missing = expected - set(wheel.namelist())
        if missing:
            raise AssertionError(f"Wheel is missing package files: {sorted(missing)}")
        stale = [name for name in sorted(expected)
                 if wheel.read(name) != (package / name.removeprefix("s1compiler/")).read_bytes()]
        if stale:
            raise AssertionError(f"Wheel contains stale package files: {stale}")
        entry_files = [name for name in wheel.namelist() if name.endswith(".dist-info/entry_points.txt")]
        assert len(entry_files) == 1, "Expected one wheel entry-point manifest"
        entries = ConfigParser()
        entries.read_string(wheel.read(entry_files[0]).decode("utf-8"))
        assert entries["console_scripts"]["s1"] == "s1compiler.cli:main"
        assert entries["console_scripts"]["s1-study"] == "s1compiler.hierarchy_study:main"
    print(f"Verified {len(expected)} library/starter files and both console entry points")


if __name__ == "__main__":
    main()
