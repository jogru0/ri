use std::fmt::Display;

use assert_cmd::Command;

fn run(file: &str, expected: impl Display) {
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
    run("res/aoc/7s.ri", 2437272016585i64);
}

#[test]
#[ignore = "slow"]
fn aoc7g() {
    run("res/aoc/7g.ri", 162987117690649i64);
}

#[test]
fn aoc8s() {
    run("res/aoc/8s.ri", 273);
}

#[test]
fn aoc8g() {
    run("res/aoc/8g.ri", 1017);
}

#[test]
fn aoc9s() {
    run("res/aoc/9s.ri", 6370402949053i64);
}

#[test]
#[ignore = "slow"]
fn aoc9g() {
    run("res/aoc/9g.ri", 6398096697992i64);
}

#[test]
fn aoc10s() {
    run("res/aoc/10s.ri", 489);
}

#[test]
fn aoc10g() {
    run("res/aoc/10g.ri", 1086);
}

#[test]
fn aoc11s() {
    run("res/aoc/11s.ri", 235850);
}

#[test]
fn aoc11g() {
    run("res/aoc/11g.ri", 279903140844645i64);
}

#[test]
fn aoc12s() {
    run("res/aoc/12s.ri", 1522850);
}

#[test]
fn aoc12g() {
    run("res/aoc/12g.ri", 953738);
}

#[test]
fn aoc13s() {
    run("res/aoc/13s.ri", 35082);
}

#[test]
fn aoc13g() {
    run("res/aoc/13g.ri", 82570698600470i64);
}

#[test]
fn aoc14s() {
    run("res/aoc/14s.ri", 230436441);
}

#[test]
fn aoc14g() {
    run("res/aoc/14g.ri", 8270);
}

#[test]
fn aoc15s() {
    run("res/aoc/15s.ri", 1426855);
}

#[test]
fn aoc15g() {
    run("res/aoc/15g.ri", 1404917);
}

#[test]
fn aoc16s() {
    run("res/aoc/16s.ri", 99488);
}

#[test]
fn aoc16g() {
    run("res/aoc/16g.ri", 516);
}

#[test]
fn aoc17s() {
    run("res/aoc/17s.ri", "[1, 5, 7, 4, 1, 6, 0, 3, 0, ]");
}

#[test]
#[ignore = "slow"]
fn aoc17g() {
    run("res/aoc/17g.ri", 108107574778365i64);
}

#[test]
fn aoc18s() {
    run("res/aoc/18s.ri", 308);
}

#[test]
#[ignore = "slow"]
fn aoc18g() {
    run("res/aoc/18g.ri", "[46, 28, ]");
}

#[test]
fn aoc19s() {
    run("res/aoc/19s.ri", 240);
}

#[test]
fn aoc19g() {
    run("res/aoc/19g.ri", 848076019766013i64);
}

#[test]
fn aoc20s() {
    run("res/aoc/20s.ri", 1358);
}

#[test]
#[ignore = "slow"]
fn aoc20g() {
    run("res/aoc/20g.ri", 1005856);
}

#[test]
fn aoc21s() {
    run("res/aoc/21s.ri", 184718);
}

#[test]
fn aoc21g() {
    run("res/aoc/21g.ri", 228800606998554i64);
}

#[test]
fn aoc22s() {
    run("res/aoc/22s.ri", 19927218456i64);
}

#[test]
#[ignore = "slow"]
fn aoc22g() {
    run("res/aoc/22g.ri", 2189);
}
