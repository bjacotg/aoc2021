use clap::Parser;
use regex::Regex;
use std::{fs, panic};

#[derive(Parser)]
#[clap(version = "1.0", author = "Kevin K. <kbknapp@gmail.com>")]
struct Opts {
    #[clap(short, long)]
    problem: i32,
}

fn main() {
    let opts = Opts::parse();
    let file = format!("input/{}.txt", opts.problem);
    let contents = fs::read_to_string(file).expect("Unable to read file");
    let function = match opts.problem {
        1 => func1,
        2 => func2,
        3 => func3,
        _ => panic!("Unexpected number provided"),
    };

    let (first, second) = function(&contents);

    println!("Result first {:?} second {:?}", first, second);
}

type Output = (Option<usize>, Option<usize>);

fn func1(input: &str) -> Output {
    let measurement = input
        .lines()
        .map(|s| s.parse::<i32>().unwrap())
        .collect::<Vec<_>>();
    let increasing_count = |seq: &[_]| {
        seq.iter()
            .zip(seq.iter().skip(1))
            .filter(|(f, s)| f < s)
            .count()
    };
    let first = increasing_count(&measurement);
    let sliding_sum = {
        let i1 = measurement.iter();
        let i2 = measurement.iter().skip(1);
        let i3 = measurement.iter().skip(2);
        i1.zip(i2)
            .zip(i3)
            .map(|((i, j), k)| i + j + k)
            .collect::<Vec<_>>()
    };
    let second = increasing_count(&sliding_sum);

    let second_smarter = measurement
        .iter()
        .zip(measurement.iter().skip(3))
        .filter(|(f, s)| f < s)
        .count();
    assert_eq!(second, second_smarter);

    (Some(first), Some(second))
}

fn func2(input: &str) -> Output {
    enum Direction {
        Forward(i32),
        Down(i32),
        Up(i32),
    }
    let re = Regex::new(r"^([a-z]+) (\d+)$").expect("Bad regex");
    let instructions = input
        .lines()
        .map(|instruction| {
            let cap = re.captures(instruction).expect("No match");
            let count = cap[2].parse().unwrap();
            match &cap[1] {
                "down" => Direction::Down(count),
                "forward" => Direction::Forward(count),
                "up" => Direction::Up(count),
                _ => panic!("Unexpected instruction: {}", &cap[1]),
            }
        })
        .collect::<Vec<_>>();

    let end_position = instructions
        .iter()
        .fold((0, 0), |(hpos, depth), instruction| match instruction {
            Direction::Forward(i) => (hpos + i, depth),
            Direction::Down(i) => (hpos, depth + i),
            Direction::Up(i) => (hpos, depth - i),
        });
    let actual_position =
        instructions.iter().fold(
            (0, 0, 0),
            |(hpos, depth, aim), instruction| match instruction {
                Direction::Forward(i) => (hpos + i, depth + i * aim, aim),
                Direction::Down(i) => (hpos, depth, aim + i),
                Direction::Up(i) => (hpos, depth, aim - i),
            },
        );

    (
        Some((end_position.0 * end_position.1) as usize),
        Some((actual_position.0 * actual_position.1) as usize),
    )
}

fn func3(input: &str) -> Output {
    let lines = input.lines().collect::<Vec<_>>();
    let count = {
        let mut count = vec![0; lines[0].len()];
        for element in &lines {
            element
                .chars()
                .zip(count.iter_mut())
                .for_each(|(e, c)| *c += if e == '1' { 1 } else { 0 });
        }
        count
    };
    let half_point = lines.len() as i32 / 2;
    let gamma = count
        .iter()
        .map(|i| if i > &half_point { 1 } else { 0 })
        .fold(0, |acc, bit| 2 * acc + bit);
    let epsilon = count
        .iter()
        .map(|i| if i > &half_point { 0 } else { 1 })
        .fold(0, |acc, bit| 2 * acc + bit);

    let oxy = {
        let mut potential = lines.clone();
        for index in 0..lines[0].len() {
            if potential.len() == 1 {
                break;
            }
            let total_count = potential.len();
            let one_count = potential.iter().filter(|s|s.as_bytes()[index] as char == '1').count();
            potential = potential
                .into_iter()
                .filter(|p| {
                    p.as_bytes()[index] as char == if 2*one_count >= total_count { '1' } else { '0' }
                })
                .collect();
        }
        potential[0].chars().fold(0,|acc, bit| 2* acc + if bit == '1' {1} else {0})
    };
 
    let co2 = {
        let mut potential = lines.clone();
        for index in 0..lines[0].len() {
            if potential.len() == 1 {
                break;
            }
            let total_count = potential.len();
            let one_count = potential.iter().filter(|s|s.as_bytes()[index] as char == '1').count();
            potential = potential
                .into_iter()
                .filter(|p| {
                    p.as_bytes()[index] as char == if 2*one_count < total_count { '1' } else { '0' }
                })
                .collect();
        }
        potential[0].chars().fold(0,|acc, bit| 2* acc + if bit == '1' {1} else {0})
    };
    (Some(gamma * epsilon), Some(oxy * co2))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn problem1() {
        let input = "1
        2
        3
        4
        1
        2
        1
        2"
        .replace(" ", "");
        assert_eq!(func1(&input), (Some(5), Some(2)));
    }

    #[test]
    fn problem2() {
        let input = "forward 5\ndown 5\nforward 8\nup 3\ndown 8\nforward 2";
        assert_eq!(func2(input), (Some(150), Some(900)));
    }

    #[test]
    fn problem3() {
        let input = "00100
        11110
        10110
        10111
        10101
        01111
        00111
        11100
        10000
        11001
        00010
        01010"
            .replace(" ", "");
        assert_eq!(func3(&input), (Some(198), Some(230)));
    }
}
