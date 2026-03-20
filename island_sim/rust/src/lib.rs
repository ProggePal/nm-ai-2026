/// PyO3 bindings for island_sim_core.

mod types;
mod params;
mod world;
mod engine;
mod phases;

use numpy::{PyArray3, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyArray1, PyArrayMethods};
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

    // Create 1D array then reshape to 3D
    let arr1d = PyArray1::from_vec_bound(py, flat);
    let arr3d = arr1d.reshape([h, w, types::NUM_CLASSES])
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("reshape failed: {}", e)))?;

    Ok(arr3d)
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
        });
    }

    Ok(WorldState {
        grid: rust_grid,
        settlements: rust_settlements,
        width,
        height,
    })
}

/// Convert Rust grid back to numpy 2D array.
fn array2d_to_numpy<'py>(py: Python<'py>, grid: &[Vec<i32>], height: usize, width: usize) -> Bound<'py, PyArray2<i32>> {
    let mut flat = Vec::with_capacity(height * width);
    for row in grid {
        flat.extend_from_slice(row);
    }
    let arr1d = PyArray1::from_vec_bound(py, flat);
    arr1d.reshape([height, width]).unwrap()
}

/// Convert Rust settlements to Python list of dicts.
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

#[pymodule]
fn island_sim_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate, m)?)?;
    m.add_function(wrap_pyfunction!(generate_prediction, m)?)?;
    Ok(())
}
