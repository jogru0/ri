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

#[test]
fn ex2() {
    let assert = run("res/examples/ex2.ri");

    assert
        .success()
        .code(0)
        .stdout("Result: 70492524767089125814114\n");
}

#[test]
fn ex3() {
    let assert = run("res/examples/ex3.ri");

    assert.success().code(0).stdout("Result: 3\n");
}

#[test]
fn ex4() {
    let assert = run("res/examples/ex4.ri");

    assert.success().code(0).stdout("Result: 2\n");
}

#[test]
fn ex5() {
    let assert = run("res/examples/ex5.ri");

    assert.success().code(0).stdout("Result: 2\n");
}

#[test]
fn ex6() {
    let assert = run("res/examples/ex6.ri");

    assert.success().code(0).stdout("Result: true\n");
}

#[test]
fn ex7() {
    let assert = run("res/examples/ex7.ri");

    assert
        .success()
        .code(0)
        .stdout("Result: [30, -1, 7, 6, (), 7, (), -1, (), ]\n");
}

//TODO!!!!!!!!!!!!!!
#[test]
fn ex8() {
    let assert = run("res/examples/ex8.ri");

    assert.success().code(0).stdout("Result: -1\n");
}
