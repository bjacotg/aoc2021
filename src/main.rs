use std::{panic, fs};

use clap::{Parser};

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
       _ => panic!("Unexpected number provided")
   };


     let (first, second) = function(&contents);
    
    println!("Result first {:?} second {:?}", first, second);
}

type Output = (Option<usize>, Option<usize>);

fn func1(input: &str) -> Output {
    let measurement = input.lines().map(|s|s.parse::<i32>().unwrap()).collect::<Vec<_>>();
    let increasing_count = |seq: &[_]| seq.iter().zip(seq.iter().skip(1)).filter(|(f,s)|f<s).count();
    let first = increasing_count(&measurement);
    let sliding_sum = {
        let i1 = measurement.iter();
        let i2 = measurement.iter().skip(1);
        let i3 = measurement.iter().skip(2);
        i1.zip(i2).zip(i3).map(|((i, j), k)|i+j+k).collect::<Vec<_>>()
    };
    let second = increasing_count(&sliding_sum);

    let second_smarter = measurement.iter().zip(measurement.iter().skip(3)).filter(|(f,s)|f < s).count();
    assert_eq!(second, second_smarter);

    (Some(first), Some(second))
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
        2".replace(" ", "");
        assert_eq!(func1(&input), (Some(5), Some(2)));
    }
}

