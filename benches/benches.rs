extern crate forest_2;

use forest_2::RankVector;


#[macro_use]
extern crate bencher;
extern crate rand;

use bencher::Bencher;
use rand::thread_rng;
use rand::Rng;
use rand::distributions::Standard;
use rand::seq::sample_indices;

fn random_vector() -> Vec<f64> {

    let mut out: Vec<f64> = thread_rng().sample_iter(&Standard).collect();

    let mut rng = thread_rng();

    for i in sample_indices(&mut rng,out.len(),30) {
        out[i] = 0.;
    }

    out

}

fn random_fill_test() {
    if random_vector().into_iter().sum::<f64>() < 0. {
        panic!();
    }
}


// fn bench_link(bench: &mut Bencher) {
//     random_fill_test();
//     let rv = random_vector();
//     bench.iter(|| {
//         RankVector::link(rv);
//     });
//
// }

benchmark_group!(benches,random_vector);
benchmark_main!(benches);
