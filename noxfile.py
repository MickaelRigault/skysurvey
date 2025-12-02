import nox

nox.options.reuse_venv = "yes"


@nox.session()
def tests(session):
    session.install(".[tests]")
    session.run("pytest", *session.posargs)


@nox.session()
def coverage(session):
    session.install(".[tests]")
    session.run("coverage", "run", "-m", "pytest", *session.posargs)
    session.run("coverage", "report")


@nox.session
def docs(session):
    session.install(".[docs]")
    with session.chdir("docs"):
        # fmt: off
        session.run(
            "python", "-m", "sphinx",
            "-T", "-E", "--keep-going",
            "-b", "html",
            "-d", "_build/doctrees",
            "-j", "auto",
            ".",
            "_build/html",
        )
        # fmt: on
