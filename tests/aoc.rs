use assert_cmd::Command;

fn run(file: &str, expected: i128) {
    let mut cmd = Command::cargo_bin("ri").unwrap();

    cmd.arg(file)
        .assert()
        .success()
        .code(0)
        .stdout(format!("Result: {expected}\n"));
}

#[test]
fn aoc1s() {
    run("res/aoc/1s.ri", 1223326);
}

#[test]
fn aoc1g() {
    run("res/aoc/1g.ri", 21070419);
}

#[test]
fn aoc2s() {
    run("res/aoc/2s.ri", 559);
}

#[test]
fn aoc2g() {
    run("res/aoc/2g.ri", 601);
}

#[test]
fn aoc3s() {
    run("res/aoc/3s.ri", 165225049);
}

#[test]
fn aoc3g() {
    run("res/aoc/3g.ri", 108830766);
}

#[test]
fn aoc4s() {
    run("res/aoc/4s.ri", 2639);
}

#[test]
fn aoc4g() {
    run("res/aoc/4g.ri", 2005);
}

#[test]
fn aoc5s() {
    run("res/aoc/5s.ri", 5064);
}

#[test]
fn aoc5g() {
    run("res/aoc/5g.ri", 5152);
}

#[test]
fn aoc6s() {
    run("res/aoc/6s.ri", 5534);
}

#[test]
#[ignore = "slow"]
fn aoc6g() {
    run("res/aoc/6g.ri", 2262);
}

#[test]
fn aoc7s() {
    run("res/aoc/7s.ri", 2437272016585);
}

#[test]
#[ignore = "slow"]
fn aoc7g() {
    run("res/aoc/7g.ri", 162987117690649);
}

#[test]
fn aoc8s() {
    run("res/aoc/8s.ri", 273);
}

#[test]
fn aoc8g() {
    run("res/aoc/8g.ri", 1017);
}
