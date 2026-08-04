#pragma once

#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/preconditioner/runtime.hpp>
#include <amgcl/solver/runtime.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

typedef amgcl::backend::builtin< double > Backend;

typedef amgcl::make_solver< amgcl::runtime::preconditioner< Backend >, amgcl::runtime::solver::wrapper< Backend > >
  Solver;

class LinearSolver {
public:
  boost::property_tree::ptree prm;

  // Cached solver and matrix structure information
  std::unique_ptr< Solver > solver_;
  int                       cached_n;
  int                       cached_nnz;
  std::vector< int >        cached_ptr_;
  std::vector< int >        cached_col_;

  // Near null-space vectors, kept alive here because the property tree only stores a raw pointer to
  // them (AMGCL copies from it when the hierarchy is built). Must outlive every solver construction.
  std::vector< double > nullspace_;

  // Constructor: Just stores the parameters
  LinearSolver( const char* json_params ) : solver_(), cached_n( -1 ), cached_nnz( -1 )
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

  void solve( int           n,
              const int*    ptr,
              const int*    col,
              const double* val,
              const double* rhs,
              double*       x,
              int&          iters,
              double&       error )
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

    std::tie( iters, error ) = ( *solver_ )( amgcl::make_iterator_range( rhs, rhs + n ),
                                             amgcl::make_iterator_range( x, x + n ) );
  }
};
