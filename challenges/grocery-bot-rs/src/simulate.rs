mod engine;
mod planner;
mod scenario;
mod state;
mod types;

use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "simulate")]
#[command(about = "Run the grocery bot against a recorded game scenario locally")]
struct Args {
    /// Path to a recording file
    #[arg(long)]
    recording: PathBuf,

    /// Print state each round (verbose)
    #[arg(long)]
    verbose: bool,
}

fn main() {
    let args = Args::parse();

    let scenario = match scenario::load_scenario(&args.recording) {
        Ok(s) => s,
        Err(err) => {
            eprintln!("Error: {err}");
            std::process::exit(1);
        }
    };

    eprintln!(
        "Scenario: {} ({}x{} grid, {} bots, {} items, {} orders, {} max rounds)",
        scenario.difficulty,
        scenario.grid.width,
        scenario.grid.height,
        scenario.bots.len(),
        scenario.items.len(),
        scenario.orders.len(),
        scenario.max_rounds,
    );

    let mut engine = engine::Engine::new(&scenario);
    let verbose = args.verbose;

    let result = engine.run(|state| {
        if verbose && state.round % 50 == 0 {
            eprintln!(
                "  Round {}/{} | Score: {} | Items on map: {} | Bot positions: {:?}",
                state.round,
                state.max_rounds,
                state.score,
                state.items.len(),
                state.bots.iter().map(|b| b.position).collect::<Vec<_>>(),
            );
        }
        planner::plan_round(state)
    });

    eprintln!();
    eprintln!("=== Simulation Complete ===");
    eprintln!("Score:            {}", result.score);
    eprintln!("Items delivered:  {}", result.items_delivered);
    eprintln!("Orders completed: {}", result.orders_completed);
    eprintln!("Rounds used:      {}/{}", result.rounds_used, scenario.max_rounds);

    // Print score to stdout for easy parsing by scripts/agents
    println!("{}", result.score);
}
