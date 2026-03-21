/// PyO3 bindings for island_sim_core.

mod types;
mod params;
mod world;
mod engine;
mod phases;
mod scoring;
mod postprocess;
mod grid_search;
mod cmaes_inference;

use numpy::{PyArray3, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict};

use types::{WorldState, Settlement};
use params::SimParams;

/// Run one simulation. Returns (grid_2d, settlements_list).
#[pyfunction]
fn simulate<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray2<'py, i32>,
    settlements: &Bound<'py, PyList>,
    params_array: PyReadonlyArray1<'py, f64>,
    sim_seed: u64,
) -> PyResult<(Bound<'py, PyArray2<i32>>, Bound<'py, PyList>)> {
    let state = parse_world_state(&grid, settlements)?;
    let params = SimParams::from_array(params_array.as_slice()?);

    let result = engine::simulate(&state, &params, sim_seed);

    let result_grid = array2d_to_numpy(py, &result.grid, result.height as usize, result.width as usize);
    let result_settlements = settlements_to_py(py, &result.settlements)?;

    Ok((result_grid, result_settlements))
}

/// Run Monte Carlo and return (H, W, 6) probability tensor.
#[pyfunction]
fn generate_prediction<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray2<'py, i32>,
    settlements: &Bound<'py, PyList>,
    params_array: PyReadonlyArray1<'py, f64>,
    num_runs: usize,
    base_seed: u64,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let state = parse_world_state(&grid, settlements)?;
    let params = SimParams::from_array(params_array.as_slice()?);

    let flat = engine::generate_prediction(&state, &params, num_runs, base_seed);

    let h = state.height as usize;
    let w = state.width as usize;

    let arr1d = PyArray1::from_vec_bound(py, flat);
    let arr3d = arr1d.reshape([h, w, types::NUM_CLASSES])
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("reshape failed: {}", e)))?;

    Ok(arr3d)
}

/// Full Rust grid search. Accepts ground truth data, returns top N results as JSON string.
///
/// Args:
///   rounds: list of lists of (grid_2d, settlements_list, ground_truth_3d) tuples
///   num_candidates: number of candidate param sets
///   mc_runs: MC runs per seed per candidate
///   top_n: number of top results to return
///
/// Returns: list of dicts with keys: params (list[float]), mean_score (float), per_round_scores (list[float])
#[pyfunction]
fn run_grid_search<'py>(
    py: Python<'py>,
    rounds: &Bound<'py, PyList>,
    num_candidates: usize,
    mc_runs: usize,
    top_n: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyList>> {
    // Parse all round data from Python
    let mut rust_rounds: Vec<Vec<grid_search::SeedData>> = Vec::new();

    for round_item in rounds.iter() {
        let round_list: &Bound<'py, PyList> = round_item.downcast()?;
        let mut seeds = Vec::new();

        for seed_item in round_list.iter() {
            let seed_tuple: &Bound<'py, pyo3::types::PyTuple> = seed_item.downcast()?;

            let grid: PyReadonlyArray2<'py, i32> = seed_tuple.get_item(0)?.extract()?;
            let settlements_obj = seed_tuple.get_item(1)?;
            let settlements: &Bound<'py, PyList> = settlements_obj.downcast()?;
            let gt: PyReadonlyArray3<'py, f64> = seed_tuple.get_item(2)?.extract()?;

            let state = parse_world_state(&grid, settlements)?;

            let gt_arr = gt.as_array();
            let height = gt_arr.shape()[0];
            let width = gt_arr.shape()[1];
            let mut gt_flat = vec![0.0_f64; height * width * types::NUM_CLASSES];
            for y in 0..height {
                for x in 0..width {
                    for c in 0..types::NUM_CLASSES {
                        gt_flat[(y * width + x) * types::NUM_CLASSES + c] = gt_arr[[y, x, c]];
                    }
                }
            }

            seeds.push(grid_search::SeedData {
                initial_state: state,
                ground_truth: gt_flat,
                height,
                width,
            });
        }
        rust_rounds.push(seeds);
    }

    // Release GIL during computation
    let results = py.allow_threads(|| {
        grid_search::run_grid_search(&rust_rounds, num_candidates, mc_runs, top_n, seed)
    });

    // Convert results back to Python
    let result_list = PyList::empty_bound(py);
    for r in &results {
        let dict = PyDict::new_bound(py);
        let params_list = PyList::new_bound(py, &r.params);
        let scores_list = PyList::new_bound(py, &r.per_round_scores);
        dict.set_item("params", params_list)?;
        dict.set_item("mean_score", r.mean_score)?;
        dict.set_item("per_round_scores", scores_list)?;
        result_list.append(dict)?;
    }

    Ok(result_list)
}

/// Parse Python WorldState into Rust.
fn parse_world_state(
    grid: &PyReadonlyArray2<'_, i32>,
    settlements: &Bound<'_, PyList>,
) -> PyResult<WorldState> {
    let grid_arr = grid.as_array();
    let height = grid_arr.shape()[0] as i32;
    let width = grid_arr.shape()[1] as i32;

    let mut rust_grid = vec![vec![0i32; width as usize]; height as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            rust_grid[y][x] = grid_arr[[y, x]];
        }
    }

    let mut rust_settlements = Vec::new();
    for item in settlements.iter() {
        let x: i32 = item.getattr("x")?.extract()?;
        let y: i32 = item.getattr("y")?.extract()?;
        let population: f64 = item.getattr("population")?.extract()?;
        let food: f64 = item.getattr("food")?.extract()?;
        let wealth: f64 = item.getattr("wealth")?.extract()?;
        let defense: f64 = item.getattr("defense")?.extract()?;
        let tech_level: f64 = item.getattr("tech_level")?.extract()?;
        let has_port: bool = item.getattr("has_port")?.extract()?;
        let has_longship: bool = item.getattr("has_longship")?.extract()?;
        let owner_id: i32 = item.getattr("owner_id")?.extract()?;
        let alive: bool = item.getattr("alive")?.extract()?;

        rust_settlements.push(Settlement {
            x, y, population, food, wealth, defense, tech_level,
            has_port, has_longship, owner_id, alive,
            raid_damage_taken: 0.0,
        });
    }

    Ok(WorldState {
        grid: rust_grid,
        settlements: rust_settlements,
        width,
        height,
        war_pairs: std::collections::HashSet::new(),
    })
}

fn array2d_to_numpy<'py>(py: Python<'py>, grid: &[Vec<i32>], height: usize, width: usize) -> Bound<'py, PyArray2<i32>> {
    let mut flat = Vec::with_capacity(height * width);
    for row in grid {
        flat.extend_from_slice(row);
    }
    let arr1d = PyArray1::from_vec_bound(py, flat);
    arr1d.reshape([height, width]).unwrap()
}

fn settlements_to_py<'py>(py: Python<'py>, settlements: &[Settlement]) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty_bound(py);
    for s in settlements {
        let dict = PyDict::new_bound(py);
        dict.set_item("x", s.x)?;
        dict.set_item("y", s.y)?;
        dict.set_item("population", s.population)?;
        dict.set_item("food", s.food)?;
        dict.set_item("wealth", s.wealth)?;
        dict.set_item("defense", s.defense)?;
        dict.set_item("tech_level", s.tech_level)?;
        dict.set_item("has_port", s.has_port)?;
        dict.set_item("has_longship", s.has_longship)?;
        dict.set_item("owner_id", s.owner_id)?;
        dict.set_item("alive", s.alive)?;
        list.append(dict)?;
    }
    Ok(list)
}

/// Run CMA-ES parameter inference.
///
/// Args:
///   observations: list of dicts with keys: seed_index, viewport_x, viewport_y, viewport_w, viewport_h, grid (2D array)
///   initial_states: list of (grid_2d, settlements_list) tuples
///   mc_runs: MC runs per evaluation
///   max_evals: maximum number of function evaluations
///   population_size: CMA-ES population size (0 = auto)
///   sigma0: initial step size in normalized [0,1] space
///   warm_start: initial parameter array (or empty array for defaults)
///   seed: random seed for CMA-ES
///
/// Returns: (best_params_array, best_loss)
#[pyfunction]
fn run_cmaes_inference<'py>(
    py: Python<'py>,
    observations: &Bound<'py, PyList>,
    initial_states: &Bound<'py, PyList>,
    mc_runs: usize,
    max_evals: usize,
    population_size: usize,
    sigma0: f64,
    warm_start: PyReadonlyArray1<'py, f64>,
    seed: u64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64)> {
    // Parse initial states
    let mut states: Vec<types::WorldState> = Vec::new();
    for state_item in initial_states.iter() {
        let state_tuple: &Bound<'py, pyo3::types::PyTuple> = state_item.downcast()?;
        let grid: PyReadonlyArray2<'py, i32> = state_tuple.get_item(0)?.extract()?;
        let settlements_obj = state_tuple.get_item(1)?;
        let settlements: &Bound<'py, PyList> = settlements_obj.downcast()?;
        states.push(parse_world_state(&grid, settlements)?);
    }

    // Parse observations and group by seed_index
    let num_seeds = states.len();
    let mut obs_by_seed: Vec<Vec<cmaes_inference::ObservationData>> = (0..num_seeds).map(|_| Vec::new()).collect();

    for obs_item in observations.iter() {
        let obs_dict: &Bound<'py, PyDict> = obs_item.downcast()?;

        let seed_index: usize = obs_dict.get_item("seed_index")?.unwrap().extract()?;
        let viewport_x: usize = obs_dict.get_item("viewport_x")?.unwrap().extract()?;
        let viewport_y: usize = obs_dict.get_item("viewport_y")?.unwrap().extract()?;
        let viewport_w: usize = obs_dict.get_item("viewport_w")?.unwrap().extract()?;
        let viewport_h: usize = obs_dict.get_item("viewport_h")?.unwrap().extract()?;
        let grid: PyReadonlyArray2<'py, i32> = obs_dict.get_item("grid")?.unwrap().extract()?;
        let grid_arr = grid.as_array();

        // Convert terrain codes to class indices
        let mut terrain_classes = Vec::with_capacity(viewport_h * viewport_w);
        for row in 0..viewport_h {
            for col in 0..viewport_w {
                let terrain = grid_arr[[row, col]];
                terrain_classes.push(types::terrain_to_class(terrain));
            }
        }

        if seed_index >= num_seeds {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("observation seed_index {} >= num initial_states {}", seed_index, num_seeds)
            ));
        }
        obs_by_seed[seed_index].push(cmaes_inference::ObservationData {
            viewport_x,
            viewport_y,
            viewport_w,
            viewport_h,
            terrain_classes,
        });
    }

    // Build SeedState vec (only seeds that have observations)
    let mut seed_states: Vec<cmaes_inference::SeedState> = Vec::new();
    for (idx, obs_list) in obs_by_seed.into_iter().enumerate() {
        if !obs_list.is_empty() {
            seed_states.push(cmaes_inference::SeedState {
                initial_state: states[idx].clone(),
                observations: obs_list,
                seed_index: idx,
            });
        }
    }

    // Parse warm start
    let warm_start_slice = warm_start.as_slice()?;
    let warm_start_opt = if warm_start_slice.is_empty() {
        None
    } else {
        Some(warm_start_slice)
    };

    let pop_size = if population_size == 0 { None } else { Some(population_size) };

    // Release GIL and run CMA-ES
    let (best_params, best_loss) = py.allow_threads(|| {
        cmaes_inference::run_cmaes_inference(
            &seed_states,
            mc_runs,
            max_evals,
            pop_size,
            sigma0,
            warm_start_opt,
            seed,
        )
    });

    let result_arr = PyArray1::from_vec_bound(py, best_params);
    Ok((result_arr, best_loss))
}

/// Return (lower_bounds, upper_bounds) as numpy arrays — for testing bounds sync.
#[pyfunction]
fn get_param_bounds<'py>(py: Python<'py>) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let lower = params::bounds_lower();
    let upper = params::bounds_upper();
    (PyArray1::from_vec_bound(py, lower), PyArray1::from_vec_bound(py, upper))
}

/// Return default parameter values as numpy array — for testing.
#[pyfunction]
fn get_default_params<'py>(py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec_bound(py, params::default_params())
}

#[pymodule]
fn island_sim_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate, m)?)?;
    m.add_function(wrap_pyfunction!(generate_prediction, m)?)?;
    m.add_function(wrap_pyfunction!(run_grid_search, m)?)?;
    m.add_function(wrap_pyfunction!(run_cmaes_inference, m)?)?;
    m.add_function(wrap_pyfunction!(get_param_bounds, m)?)?;
    m.add_function(wrap_pyfunction!(get_default_params, m)?)?;
    Ok(())
}
