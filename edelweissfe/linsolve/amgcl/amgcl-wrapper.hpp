#pragma once

#include <algorithm>
#include <amgcl/adapter/block_matrix.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/preconditioner/runtime.hpp>
#include <amgcl/solver/runtime.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

// Templated on the AMGCL backend's value type, so the same wrapper serves both the default
// double-precision hierarchy and a float32 one (half the memory traffic in the smoother apply, the
// dominant cost on large coupled solves -- see PERF_LINSOLVE_INVESTIGATION.md §18/§19.3). The outer
// Krylov solve (blockamg's GMRES) always stays double; only the preconditioner's own storage and
// arithmetic narrow. rhs/x at the applyPreconditioner()/solve() boundary are always double -- the
// value-type-dependent scratch conversion happens inside this class, not at the Cython/Python
// boundary, since applyPreconditioner() is called once per outer Krylov iteration (hot path) while
// build() is not.
template < typename ValueType >
class LinearSolverT {
public:
  typedef amgcl::backend::builtin< ValueType > Backend;
  typedef amgcl::make_solver< amgcl::runtime::preconditioner< Backend >, amgcl::runtime::solver::wrapper< Backend > >
    Solver;

  boost::property_tree::ptree prm;

  // Cached solver and matrix structure information
  std::unique_ptr< Solver > solver_;
  int                       cached_n;
  int                       cached_nnz;
  std::vector< int >        cached_ptr_;
  std::vector< int >        cached_col_;

  // Near null-space vectors, kept alive here because the property tree only stores a raw pointer to
  // them (AMGCL copies from it when the hierarchy is built). Must outlive every solver construction.
  // Always double, independent of ValueType/Backend: AMGCL's coarsening::nullspace_params::B is
  // hardcoded std::vector<double> (amgcl/coarsening/tentative_prolongation.hpp) -- the tentative
  // prolongation's QR factorization stays double-precision regardless of backend, since it runs once
  // at hierarchy build time, not in the per-iteration smoother apply this backend split targets.
  std::vector< double > nullspace_;

  // Constructor: Just stores the parameters
  LinearSolverT( const char* json_params ) : solver_(), cached_n( -1 ), cached_nnz( -1 )
  {
    std::string json_str( json_params );
    if ( !json_str.empty() ) {
      std::stringstream ss( json_str );
      boost::property_tree::read_json( ss, prm );
    }
  }

  // Supply near null-space vectors for smoothed-aggregation coarsening. B is a rows-by-cols matrix in
  // row-major order (column j is the j-th near null-space vector). AMGCL takes these as a raw pointer
  // in its property tree -- they cannot travel through the JSON parameter string -- so this is a
  // separate entry point. Must be called before the first solve(); the pointer is read when the AMG
  // hierarchy is constructed. Passing cols == 0 clears any previously set null-space.
  void set_nullspace( const double* B, int rows, int cols )
  {
    if ( cols <= 0 ) {
      nullspace_.clear();
      boost::optional< boost::property_tree::ptree& > coarsening = prm.get_child_optional( "precond.coarsening" );
      if ( coarsening ) {
        coarsening->erase( "nullspace" );
      }
      return;
    }
    nullspace_.assign( B, B + static_cast< size_t >( rows ) * cols );
    prm.put( "precond.coarsening.nullspace.cols", cols );
    prm.put( "precond.coarsening.nullspace.rows", rows );
    prm.put( "precond.coarsening.nullspace.B", nullspace_.data() );
  }

  // Build the solver (and thus the AMG hierarchy) once for A, so it can be applied repeatedly as a
  // preconditioner without rebuilding. This is the build-once / apply-many split that :meth:`solve`
  // fuses: :meth:`solve` reconstructs the hierarchy on every call, which is fine for a one-shot solve
  // but ruinous for an inner block preconditioner applied on every outer Krylov iteration. Uses the
  // current property tree (including any near null-space set via set_nullspace).
  void build( int n, const int* ptr, const int* col, const ValueType* val )
  {
    int  nnz = ptr[n];
    auto A   = std::make_tuple( n,
                              amgcl::make_iterator_range( ptr, ptr + n + 1 ),
                              amgcl::make_iterator_range( col, col + nnz ),
                              amgcl::make_iterator_range( val, val + nnz ) );
    solver_.reset( new Solver( A, prm ) );
    cached_n   = n;
    cached_nnz = nnz;
    cached_ptr_.assign( ptr, ptr + n + 1 );
    cached_col_.assign( col, col + nnz );
  }

  // Apply one preconditioner (AMG) cycle to rhs: x <- M^-1 rhs, where M is the hierarchy built by
  // :meth:`build`. This is the operation a block Gauss-Seidel / field-split preconditioner performs on
  // each field per outer iteration. Cheap relative to the build. rhs/x are double regardless of
  // ValueType -- for the float backend this narrows on the way in and widens on the way out; for the
  // (default) double backend the extra copy is a no-op-equivalent memcpy, not a behaviour change.
  void applyPreconditioner( int n, const double* rhs, double* x )
  {
    std::vector< ValueType > rhs_v( rhs, rhs + n );
    std::vector< ValueType > x_v( n, ValueType( 0 ) );
    auto                     rhs_rng = amgcl::make_iterator_range( rhs_v.data(), rhs_v.data() + n );
    auto                     x_rng   = amgcl::make_iterator_range( x_v.data(), x_v.data() + n );
    solver_->precond().apply( rhs_rng, x_rng );
    std::copy( x_v.begin(), x_v.end(), x );
  }

  // Human-readable hierarchy report (levels, operator complexity, coarse size) for the AMG built by
  // build() or solve(). Streams AMGCL's own operator<< for the preconditioner + solver. Must be
  // called after a successful build()/solve() -- throws otherwise.
  std::string report() const
  {
    if ( !solver_ ) {
      throw std::runtime_error( "report(): no hierarchy built yet -- call build() or solve() first" );
    }
    std::ostringstream oss;
    oss << *solver_;
    return oss.str();
  }

  void solve( int              n,
              const int*       ptr,
              const int*       col,
              const ValueType* val,
              const double*    rhs,
              double*          x,
              int&             iters,
              double&          error )
  {

    int nnz = ptr[n];

    auto ptr_rng = amgcl::make_iterator_range( ptr, ptr + n + 1 );
    auto col_rng = amgcl::make_iterator_range( col, col + nnz );
    auto val_rng = amgcl::make_iterator_range( val, val + nnz );

    auto A = std::make_tuple( n, ptr_rng, col_rng, val_rng );

    // (Re)build or update the cached solver depending on matrix structure
    if ( !solver_ ) {
      // First call: construct the solver and cache matrix structure
      solver_.reset( new Solver( A, prm ) );
      cached_n   = n;
      cached_nnz = nnz;
      cached_ptr_.assign( ptr, ptr + n + 1 );
      cached_col_.assign( col, col + nnz );
    }
    else if ( n != cached_n || nnz != cached_nnz || !std::equal( ptr, ptr + n + 1, cached_ptr_.begin() ) ||
              !std::equal( col, col + nnz, cached_col_.begin() ) ) {
      // Matrix structure changed: rebuild solver to preserve behavior
      solver_.reset( new Solver( A, prm ) );
      cached_n   = n;
      cached_nnz = nnz;
      cached_ptr_.assign( ptr, ptr + n + 1 );
      cached_col_.assign( col, col + nnz );
    }
    else {
      solver_.reset( new Solver( A, prm ) );
    }

    std::vector< ValueType > rhs_v( rhs, rhs + n );
    std::vector< ValueType > x_v( n, ValueType( 0 ) );
    std::tie( iters, error ) = ( *solver_ )( amgcl::make_iterator_range( rhs_v.data(), rhs_v.data() + n ),
                                             amgcl::make_iterator_range( x_v.data(), x_v.data() + n ) );
    std::copy( x_v.begin(), x_v.end(), x );
  }
};

// Block-valued backend (§20.1, B3): the per-field hierarchy stores/operates on B×B nodal blocks
// (amgcl::static_matrix<double,B,B>) instead of scalar entries. Two motivations, in confidence order:
// (i) the CSR index arrays shrink by ~B² (one column index per block instead of per scalar entry --
// §19.3 found index traffic, not values, is the larger share of hierarchy bandwidth, which is exactly
// what this attacks); (ii) block-aware smoothers (block-ILU0, block-GS) invert each node's B×B
// coupling exactly, AMGCL's own canonical recipe for vector-PDE (elasticity) operators.
//
// This is a *separate* class from LinearSolverT, not another instantiation of it, because the
// construction/apply pattern genuinely differs: the matrix arrives from Python as a plain scalar CSR
// (Cython has no reason to know about block layout), so it must be wrapped with
// amgcl::adapter::block_matrix<BlockType> before reaching the block Backend, and rhs/x must be
// reinterpreted (amgcl::backend::reinterpret_as_rhs<BlockType>, a zero-copy amgcl::reinterpret_cast
// over a same-sized contiguous double buffer -- amgcl/backend/builtin.hpp -- not a per-element
// conversion) rather than element-cast like the float backend's std::vector<ValueType> narrowing.
// LinearSolverT stays untouched to keep the validated scalar/float paths at zero regression risk.
//
// set_nullspace() is not supported here and always throws: AMGCL's own tentative-prolongation
// nullspace path self-flags as unimplemented for block value types (amgcl/coarsening/
// tentative_prolongation.hpp: "TODO: this is just a workaround to make non-scalar value types
// compile. Most probably this won't actually work.") -- not merely undocumented, upstream itself does
// not trust it. This is an accepted, measured non-loss: rigid-body near-null-space vectors do not help
// on this operator anyway (§11, §13).
template < typename BlockType >
class LinearSolverBlockT {
public:
  typedef amgcl::backend::builtin< BlockType > Backend;
  typedef amgcl::make_solver< amgcl::runtime::preconditioner< Backend >, amgcl::runtime::solver::wrapper< Backend > >
    Solver;

  static const int BlockSize = amgcl::math::static_rows< BlockType >::value;

  boost::property_tree::ptree prm;
  std::unique_ptr< Solver >   solver_;
  int                         cached_n;
  int                         cached_nnz;
  std::vector< int >          cached_ptr_;
  std::vector< int >          cached_col_;

  LinearSolverBlockT( const char* json_params ) : solver_(), cached_n( -1 ), cached_nnz( -1 )
  {
    std::string json_str( json_params );
    if ( !json_str.empty() ) {
      std::stringstream ss( json_str );
      boost::property_tree::read_json( ss, prm );
    }
  }

  void set_nullspace( const double*, int, int )
  {
    throw std::runtime_error(
      "set_nullspace() is not supported with a block-valued AMGCL backend -- AMGCL's own "
      "tentative-prolongation nullspace path is an unimplemented, self-flagged 'probably won't work' "
      "TODO for block value types, not a supported feature. Do not request a near null-space when "
      "backendBlockSize > 1." );
  }

  // n must be divisible by BlockSize (node-major DOF layout: BlockSize contiguous scalar DOFs per
  // node). amgcl::adapter::block_matrix asserts this too, but only via assert(), which -DNDEBUG (this
  // extension's build flag) compiles out -- so this check is the only one that actually runs.
  void checkBlockDivisible( int n ) const
  {
    if ( n % BlockSize != 0 ) {
      throw std::runtime_error( "block-valued AMGCL backend: n=" + std::to_string( n ) +
                                " is not divisible by the block size " + std::to_string( BlockSize ) +
                                " -- the DOF layout must be node-major with BlockSize contiguous scalar "
                                "DOFs per node." );
    }
  }

  void build( int n, const int* ptr, const int* col, const double* val )
  {
    checkBlockDivisible( n );
    int  nnz = ptr[n];
    auto A   = std::make_tuple( n,
                              amgcl::make_iterator_range( ptr, ptr + n + 1 ),
                              amgcl::make_iterator_range( col, col + nnz ),
                              amgcl::make_iterator_range( val, val + nnz ) );
    solver_.reset( new Solver( amgcl::adapter::block_matrix< BlockType >( A ), prm ) );
    cached_n   = n;
    cached_nnz = nnz;
    cached_ptr_.assign( ptr, ptr + n + 1 );
    cached_col_.assign( col, col + nnz );
  }

  void applyPreconditioner( int n, const double* rhs, double* x )
  {
    std::fill( x, x + n, 0.0 );
    std::vector< double > rhs_v( rhs, rhs + n );
    std::vector< double > x_v( x, x + n );
    auto                  rhs_rng = amgcl::backend::reinterpret_as_rhs< BlockType >( rhs_v );
    auto                  x_rng   = amgcl::backend::reinterpret_as_rhs< BlockType >( x_v );
    solver_->precond().apply( rhs_rng, x_rng );
    std::copy( x_v.begin(), x_v.end(), x );
  }

  std::string report() const
  {
    if ( !solver_ ) {
      throw std::runtime_error( "report(): no hierarchy built yet -- call build() or solve() first" );
    }
    std::ostringstream oss;
    oss << *solver_;
    return oss.str();
  }

  void solve( int           n,
              const int*    ptr,
              const int*    col,
              const double* val,
              const double* rhs,
              double*       x,
              int&          iters,
              double&       error )
  {
    checkBlockDivisible( n );
    int  nnz = ptr[n];
    auto A   = std::make_tuple( n,
                              amgcl::make_iterator_range( ptr, ptr + n + 1 ),
                              amgcl::make_iterator_range( col, col + nnz ),
                              amgcl::make_iterator_range( val, val + nnz ) );

    if ( !solver_ || n != cached_n || nnz != cached_nnz || !std::equal( ptr, ptr + n + 1, cached_ptr_.begin() ) ||
         !std::equal( col, col + nnz, cached_col_.begin() ) ) {
      solver_.reset( new Solver( amgcl::adapter::block_matrix< BlockType >( A ), prm ) );
      cached_n   = n;
      cached_nnz = nnz;
      cached_ptr_.assign( ptr, ptr + n + 1 );
      cached_col_.assign( col, col + nnz );
    }
    else {
      solver_.reset( new Solver( amgcl::adapter::block_matrix< BlockType >( A ), prm ) );
    }

    std::vector< double > rhs_v( rhs, rhs + n );
    std::vector< double > x_v( n, 0.0 );
    std::tie( iters, error ) = ( *solver_ )( amgcl::backend::reinterpret_as_rhs< BlockType >( rhs_v ),
                                             amgcl::backend::reinterpret_as_rhs< BlockType >( x_v ) );
    std::copy( x_v.begin(), x_v.end(), x );
  }
};

// The default, unchanged double-precision wrapper, and the new float32 one added for §19.3.
typedef LinearSolverT< double > LinearSolver;
typedef LinearSolverT< float >  LinearSolverFloat;

// Block-valued instantiations (§20.1): 3×3 for the pryout's 3D displacement field, 2×2 for the
// registered 2D CantileverBeamQuad4BlockAMG regression test -- one template parameter apart.
typedef LinearSolverBlockT< amgcl::static_matrix< double, 2, 2 > > LinearSolverBlock2;
typedef LinearSolverBlockT< amgcl::static_matrix< double, 3, 3 > > LinearSolverBlock3;
