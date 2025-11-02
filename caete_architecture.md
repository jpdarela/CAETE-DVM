# CAETE-DVM Model Codebase Overview (Under development. Needs update)

[⇦ Back](./README.md)

## Model Architecture and Core Functionality

The CAETE-DVM (Carbon and Ecosystem Trait-based Evaluation - Dynamic Vegetation Model) is a ecosystem model that simulates vegetation dynamics, carbon cycling, and nutrient cycling at the ecosystem level. The model uses a trait-based approach where plant communities are represented by collections of Plant Life Strategies (PLS).

## Core Design Principles

1. **Trait-Based Approach**: The model represents vegetation through Plant Life Strategies (PLS) - virtual plant prototypes defined by 17 functional traits
2. **Metacommunity Structure**: Each gridcell can contain a metacommunity composed of one or more plant communities
3. **Environmental Filtering**: The model simulates how environmental conditions filter which PLS can survive and thrive
4. **Hybrid Architecture**: Combines Python for data management and control flow with Fortran for numerical computations

## Main Components and File Structure

### 1. Core Model Classes (`caete.py`)

#### Base Classes (Inheritance Hierarchy)
- **`state_zero`**: Base class handling input/output, file paths, and basic gridcell setup
- **`climate`**: Manages climate data (temperature, precipitation, radiation, humidity, atmospheric pressure)
- **`time`**: Handles temporal data, calendar operations, and time indexing
- **`soil`**: Manages soil properties including:
  - Carbon pools (4 pools: litter 1&2, soil 1&2)
  - Nutrient pools (8 pools for N and P)
  - Water content and hydraulic properties
- **`gridcell_output`**: Manages model output allocation and data flushing

#### Main Gridcell Class
- **`grd_mt`**: The primary gridcell class that inherits from all base classes
  - Represents a single simulation unit in the model
  - Contains one metacommunity with one or more plant communities
  - Handles daily ecosystem processes and biogeochemical cycling

### 2. Community Structure (`metacommunity.py`, `community.py`)

#### Plant Life Strategy (PLS) Management
- **`pls_table`**: Interface for the global table of plant prototypes
  - Stores ~20,000+ virtual plant prototypes
  - Each PLS defined by 17 functional traits
  - Provides subsampling methods for community creation

#### Community Hierarchy
- **`community`**: Represents a collection of PLS within a gridcell
  - Contains biomass state variables (leaf, wood, root carbon)
  - Tracks plant functional diversity metrics
  - Manages PLS survival and mortality
  - Handles nutrient uptake strategies

- **`metacommunity`**: Collection of communities within a gridcell
  - Enables spatial heterogeneity within a gridcell
  - Manages community-level processes
  - Tracks functional diversity and composition

### 3. Regional Management (`region.py`)

#### Region Class
- **`region`**: Manages collections of gridcells
  - Handles parallel processing of multiple gridcells
  - Manages input/output operations
  - Coordinates multiprocessing execution
  - Provides methods for saving/loading model states

### 4. Numerical Core (Fortran Modules)

#### Daily Budget Calculation (`budget.F90`)
- **`daily_budget`**: Core subroutine that calculates daily ecosystem processes
  - Photosynthesis and respiration
  - Carbon allocation
  - Nutrient cycling
  - Water balance
  - Growth and mortality

#### Photosynthesis (`photo_par.f90`)
- Contains photosynthesis parameters and constants
- Implements C3 and C4 photosynthesis models
- Handles light limitation and CO₂ response

#### Additional Fortran Modules
- **`allocation.f90`**: Carbon and nutrient allocation algorithms
- **`productivity.f90`**: NPP calculation and growth processes
- **`soil_dec.f90`**: Soil decomposition and nutrient mineralization
- **`water.mod`**: Water cycle and evapotranspiration
- **`evap.f90`**: Evaporation and transpiration calculations

### 5. Configuration and Parameters

#### Configuration System (`config.py`)
- **`Config`**: Reads parameters from `caete.toml`
- Manages model configuration including:
  - Grid resolution and coordinate reference system
  - Community size and number of communities
  - Conversion factors for input data
  - Multiprocessing settings

#### Parameter Files
- **`parameters.py`**: Global parameters
- **`caete.toml`**: Main configuration file

### 6. Execution and Workflow (`caete_driver.py`, `worker.py`)

#### Driver Scripts
- **`caete_driver.py`**: Example execution script showing typical workflow
- **`caete_driver_CMIP6.py`**: CMIP6-specific execution script

#### Worker Functions (`worker.py`)
- **`worker`**: Static methods for different simulation phases
  - Spinup procedures
  - Transient runs
  - State saving/loading
  - Parallel execution management

### 7. Supporting Modules

#### Hydrological Processes (`hydro_caete.py`)
- **`soil_water`**: Soil water dynamics
- Hydraulic conductivity calculations
- Water retention curves
- Field capacity and wilting point management

#### JIT-Compiled Functions (`caete_jit.py`)
- Performance-critical functions using Numba
- Array manipulation utilities
- Statistical calculations (Shannon diversity, etc.)
- Community-weighted means

#### Data Processing
- **`dataframes.py`**: Data processing utilities
- **`output.py`**: Output data structures and management
- **`precompilation_data.py`**: Data preparation utilities

### 8. Plant Life Strategy Generation (`plsgen.py`)

#### PLS Table Creation
- Generates quasi-random plant prototypes
- Based on literature-derived trait relationships
- Creates functional trait combinations
- Saves tables in standardized format

## Model Execution Flow

### 1. Initialization Phase
1. Load configuration from `caete.toml`
2. Create region object with climate and soil data
3. Load PLS table (global plant prototypes)
4. Initialize gridcells with metacommunities
5. Assign initial PLS to communities

### 2. Simulation Loop (Daily Time Step)
1. **Climate Forcing**: Read daily weather data
2. **Community Loop**: For each community in the metacommunity:
   - Calculate photosynthesis and respiration
   - Determine nutrient availability and uptake
   - Apply environmental filtering
   - Update biomass pools
   - Handle PLS mortality and recruitment
3. **Soil Processes**: Update soil carbon and nutrient pools
4. **Water Balance**: Calculate evapotranspiration and soil moisture
5. **Output**: Store daily results

### 3. Community Dynamics
- **Environmental Filtering**: Remove PLS that cannot survive current conditions
- **Recruitment**: Add new PLS from global table based on environmental suitability
- **Competition**: PLS compete for resources based on their traits
- **Succession**: Community composition changes over time

## Key Model Features

### 1. Trait-Based Vegetation
- 17 functional traits per PLS including:
  - Photosynthetic parameters (Vcmax, specific leaf area)
  - Allocation patterns (leaf:wood:root ratios)
  - Nutrient acquisition strategies
  - Phenological characteristics
  - Stress tolerance parameters

### 2. Biogeochemical Cycling
- **Carbon Cycle**:
  - Photosynthesis (C3/C4 pathways)
  - Plant respiration (maintenance + growth)
  - Soil decomposition (4 carbon pools)
  - Litter production and turnover

- **Nitrogen Cycle**:
  - Multiple N pools (organic, inorganic, available)
  - N fixation and mineralization
  - Plant N uptake strategies
  - N limitation on growth

- **Phosphorus Cycle**:
  - P weathering and sorption
  - Organic P mineralization
  - P uptake mechanisms
  - P limitation effects

### 3. Water Cycle
- Soil water dynamics (2 soil layers)
- Evapotranspiration modeling
- Water stress effects on photosynthesis
- Hydraulic redistribution

### 4. Spatial and Temporal Scales
- **Spatial**: Gridcell-based (typically 0.5° resolution)
- **Temporal**: Daily time step for processes, annual for community dynamics
- **Extent**: Regional to global scales

## Technical Implementation

### 1. Performance Optimization
- **Fortran Core**: Numerical computations in optimized Fortran
- **Parallel Processing**: Multiprocessing for gridcell-level parallelization
- **JIT Compilation**: Numba for Python performance bottlenecks
- **Memory Management**: Efficient array operations and data structures

### 2. Data Management
- **Compressed Storage**: BZ2 compression for input/output files
- **State Persistence**: Complete model state can be saved/restored
- **Flexible I/O**: Multiple input formats supported
- **Metadata Tracking**: Comprehensive provenance information

### 3. Model Validation and Testing
- **Unit Tests**: Individual component testing
- **Integration Tests**: Full model execution validation
- **Benchmarking**: Performance profiling capabilities
- **Debugging Tools**: Extensive logging and error handling

## Configuration and Customization

### 1. Model Parameters (`caete.toml`)
- Adjustable community size and number
- Configurable trait ranges and distributions
- Flexible time periods and forcing scenarios
- Customizable output variables and frequencies

### 2. Experimental Design
- **Spinup Protocols**: Equilibration procedures
- **Scenario Analysis**: Climate change, land use, fertilization
- **Sensitivity Analysis**: Parameter perturbation studies
- **Factorial Experiments**: Multiple factor interactions

## Applications and Use Cases

### 1. Climate Change Research
- Vegetation response to warming
- CO₂ fertilization effects
- Ecosystem adaptation and migration
- Tipping point identification

### 2. Biogeochemical Studies
- Carbon-climate feedbacks
- Nutrient limitation patterns
- Ecosystem productivity trends
- Soil carbon dynamics

### 3. Biodiversity Research
- Functional diversity patterns
- Community assembly rules
- Environmental filtering mechanisms
- Trait-environment relationships

## Additional Core Components (Complete Analysis)

### 9. Worker Functions and Execution Control (`worker.py`)

#### Worker Class
- **`worker`**: Static methods for different execution phases
  - **`create_run_breaks()`**: Creates temporal intervals for model runs
  - **`spinup()`**: Main spinup procedure with glacial/interglacial cycles
  - **`spinup_cmip6()`**: CMIP6-specific spinup procedure
  - **`soil_pools_spinup()`**: Soil equilibration procedures
  - **`soil_pools_spinup_glacial()`**: Glacial period soil spinup
  - **`soil_pools_spinup_interglacial()`**: Interglacial period soil spinup
  - **`quit_spinup()`**: Final spinup phase without PLS recruitment
  - **`run_spinup_transfer()`**: Transfer from spinup to transient phase
  - **`transclim_run()`**: Transient climate simulation
  - **`transient_run_brk()`**: Transient run with time breaks
  - **`transient_piControl_brk()`**: Pre-industrial control run
  - **`save_state_zstd()`**: Save model state with ZSTD compression
  - **`load_state_zstd()`**: Load compressed model state

### 10. JIT-Compiled Performance Functions (`caete_jit.py`)

#### Numba-Optimized Functions
- **`process_tuple()`**: Process limitation status tuples
- **`process_tuples()`**: Batch processing of limitation data
- **`pft_area_frac()`**: Calculate PFT area fractions (32-bit)
- **`pft_area_frac64()`**: Calculate PFT area fractions (64-bit)
- **`neighbours_index()`**: Find neighboring cells in grid
- **`inflate_array()`**: Expand partial arrays to full size
- **`linear_func()`**: Linear coupling function for atmosphere-surface
- **`atm_canopy_coupling()`**: Atmosphere-canopy coupling calculation
- **`masked_mean()`**: Calculate means with masked values
- **`masked_mean_2D()`**: 2D array masked mean calculation
- **`cw_mean()`**: Community-weighted mean calculation
- **`cw_variance()`**: Community-weighted variance calculation
- **`shannon_entropy()`**: Shannon entropy for diversity
- **`shannon_evenness()`**: Shannon evenness index
- **`shannon_diversity()`**: Shannon diversity index

### 11. Plant Life Strategy Generation (`plsgen.py`)

#### PLS Table Generation System
- **`get_parameters()`**: Read parameters from plsgen.toml
- **`check_viability()`**: Validate PLS trait combinations
- **`assert_data_size()`**: Check dataset size requirements
- **`allocation_combinations()`**: Generate allocation patterns using Dirichlet distribution
- **`nutrient_ratios()`**: Generate N:C and P:C ratios
- **`carbon_coefficients()`**: Create carbon allocation and residence time combinations
- **`table_gen()`**: Main function to generate complete PLS table
- **Constants**: Grass/woody ratios, minimum allocations, residence times

### 12. Data Processing and Output (`dataframes.py`)

#### Output Management System
- **`get_spins()`**: List available model output periods
- **`print_variables()`**: Display available output variables
- **`get_var_metadata()`**: Retrieve variable metadata (units, descriptions)
- **`output_manager`**: Class with methods for:
  - **`generic_text_output_grd()`**: Generic gridded text output
  - **`cities_output()`**: City-specific output processing
  - **`region_grid_output()`**: Regional gridded output
  - **`process_multiple_regions()`**: Multi-region processing
  - **Variable definitions**: Complete metadata for 50+ output variables

### 13. Geospatial Functions (`_geos.py`)

#### Geographic Utilities
- **`calculate_area()`**: Calculate grid cell area using geodesic calculations
- **`find_indices_xy()`**: Convert lat/lon to grid indices
- **`find_coordinates_xy()`**: Convert grid indices to lat/lon
- **`get_region()`**: Extract regional boundaries
- **`pan_amazon_region`**: Predefined Amazon region coordinates
- **Unit test classes**: Comprehensive testing for geographic functions

### 14. Hydrological Processes (`hydro_caete.py`)

#### Soil Water Dynamics
- **`B_func()`**: Moisture-tension coefficient calculation
- **`ksat_func()`**: Saturated hydraulic conductivity
- **`kth_func()`**: Unsaturated hydraulic conductivity
- **`soil_water`**: Class managing soil water dynamics
  - **Water retention parameters**: Wilting point, field capacity, saturation
  - **Water pool management**: Upper and lower soil layers
  - **Available water capacity**: Calculations for plant uptake

### 15. Complete Fortran Module Analysis

#### Core Fortran Modules (`*.f90`)
- **`types.f90`**: Precision definitions (2/4/8 byte integers/floats)
- **`global_par.f90`**: Global parameters and constants
  - Q10 temperature response (1.4)
  - Soil thermal properties
  - Stomatal resistance limits
  - Number of PLS (200) and traits (17)
  - OpenMP thread settings

#### Carbon Costs Module (`cc.f90`)
- **`calc_passive_uptk1()`**: Passive nutrient uptake via transpiration
- **`passive_uptake()`**: Estimate passive N and P uptake
- **`cc_active()`**: Carbon costs of active uptake
- **`active_costN()`**: Nitrogen active uptake costs
- **`active_costP()`**: Phosphorus active uptake costs
- **`cc_fix()`**: Carbon costs of N fixation
- **`fixed_n()`**: N fixation calculations
- **`cc_retran()`**: Retranslocation costs
- **`retran_nutri_cost()`**: Nutrient retranslocation
- **`select_active_strategy()`**: Choose optimal uptake strategy
- **8 uptake strategies**: Different pathways for N and P acquisition

#### Photosynthesis Functions (`funcs.f90`)
- **`gross_ph()`**: Gross photosynthesis calculation
- **`leaf_area_index()`**: LAI calculation
- **`f_four()`**: Auxiliary function for light calculations
- **`spec_leaf_area()`**: Specific leaf area
- **`sla_reich()`**: SLA based on Reich et al.
- **`leaf_nitrogen_concentration()`**: Leaf N concentration
- **`water_stress_modifier()`**: F5 water stress function
- **`photosynthesis_rate()`**: Leaf-level CO2 assimilation
- **`vcmax_a()`**: Vcmax from Domingues et al. (2010)
- **`stomatal_resistance()`**: Canopy resistance (Medlyn et al. 2011)
- **`vapor_p_deficit()`**: Vapor pressure deficit
- **`transpiration()`**: Plant transpiration
- **`tetens()`**: Saturation vapor pressure
- **`m_resp()`**: Maintenance respiration
- **`g_resp()`**: Growth respiration
- **`realized_npp()`**: NPP after limitations
- **`spinup2()`**: Vegetation pool spinup
- **`spinup3()`**: Viability check for trait combinations
- **`water_ue()`**: Water use efficiency
- **`leap()`**: Leap year detection

### 16. Configuration and Parameter Management

#### Configuration System (`config.py`)
- **`Config`**: Nested configuration class with attribute access
- **`fetch_config()`**: Load configuration from TOML file
- **`get_fortran_runtime()`**: Windows-specific Fortran runtime path
- **`update_sys_pathlib()`**: DLL path management for Windows

#### Main Configuration (`caete.toml`)
- **Multiprocessing settings**: Number of processes and threads
- **Conversion factors**: Unit conversions for ISIMIP data
- **Metacommunity configuration**: Number of communities and PLS
- **Trait ranges**: Maximum values for PLS traits
- **Fertilization experiments**: N and P addition amounts
- **Output settings**: Variables and frequencies
- **Day-of-year settings**: Monthly environmental filtering dates

#### Parameter Definitions (`parameters.py`)
- **File paths**: Input data locations and output directories
- **Soil parameters**: Water retention curves for global grids
  - `tsoil`: Topsoil properties (saturation, field capacity, wilting point)
  - `ssoil`: Subsoil properties
  - `hsoil`: Hydraulic properties (porosity, water potential, texture)
- **Regional masks**: Pan-Amazon forest masks
- **Standard filenames**: PLS tables, state files, run identifiers

### 17. Testing and Validation Framework

#### Test Modules (`tests/`)
- **`caete_import.py`**: Module import tests
- **`canopy_resistance.py`**: Stomatal resistance testing
- **`cc_test.py`**: Carbon cost function tests
- **`century_daily.py`**: Daily soil carbon model tests
- **`ncycle.py`**: Nitrogen cycle testing
- **`test_atm_canopy_coupling.py`**: Atmosphere-canopy coupling tests
- **`test_carbon3.py`**: Soil decomposition tests
- **`test_f4.py`**: Light function tests
- **`test_n_conc.py`**: Nitrogen concentration tests
- **`test_sla.py`**: Specific leaf area tests
- **`test_spinup3.py`**: Spinup viability tests
- **`test_vcmax.py`**: Vcmax function tests
- **`water_stress_modifier.py`**: Water stress tests

### 18. Utilities and Support Modules

#### Execution Scripts
- **`from_state.py`**: Resume simulations from saved states
- **`time_output_manager.py`**: Time execution performance
- **`k34_experiment.py`**: Site-specific experiments
- **`sandbox.py`**: Development and testing script

#### Profiling and Performance
- **`profiling/`**: Performance profiling tools
- **`profile_dataframes.py`**: Output processing performance
- **Memory profiling**: Integration with memory_profiler

#### Debug and Development
- **`debug_aux/`**: Debugging utilities and helper functions
- **`equations/`**: Mathematical equation implementations
- **`windows_utils/`**: Windows-specific utilities

### 19. Model Architecture Summary

#### Data Flow Architecture
1. **Initialization**: Load PLS table → Create region → Initialize gridcells → Setup metacommunities
2. **Daily Loop**: Climate forcing → Community processes → Soil processes → Water balance → Output
3. **Community Dynamics**: Environmental filtering → PLS recruitment → Competition → Succession
4. **Annual Processing**: Diversity calculations → State saving → Community reset options

#### Process Integration
- **Photosynthesis-Allocation-Growth**: Coupled C-N-P cycling with trait-based constraints
- **Water-Carbon-Nutrients**: Integrated limitation effects on all processes
- **Competition-Succession**: Trait-based environmental filtering and community assembly
- **Spatial-Temporal**: Hierarchical organization from PLS to communities to metacommunities to regions

#### Performance Optimization
- **Fortran Core**: All numerically intensive calculations
- **Python Management**: Data handling, I/O, multiprocessing
- **JIT Compilation**: Critical Python functions optimized with Numba
- **Memory Management**: Efficient array operations and compressed storage
- **Parallel Processing**: Gridcell-level parallelization with OpenMP threads

[⇦ Back](./README.md)