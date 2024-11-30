use assert_cmd::{assert::Assert, Command};

fn run(file: &str) -> Assert {
    let mut cmd = Command::cargo_bin("ri").unwrap();

    cmd.arg(file).assert()
}

#[test]
fn ex0() {
    let assert = run("res/examples/ex0.ri");

    assert.success().code(0).stdout("Result: 55\n");
}

#[test]
fn ex0_clean() {
    let assert = run("res/examples/ex0_clean.ri");

    assert.success().code(0).stdout("Result: 55\n");
}