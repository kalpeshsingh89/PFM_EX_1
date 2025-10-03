#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>

using namespace dealii;

template <int dim>
class PhaseFieldMonolithicConsistentAMR
{
public:
  PhaseFieldMonolithicConsistentAMR();
  void run();

private:
  // problem setup
  void make_grid();
  void setup_system();
  void initialize_fields();

  // analysis loop
  void assemble_residual_and_jacobian();
  void solve_linear_system();
  void update_solution_and_enforce_bounds(const double alpha);
  void update_history_field(); // H <- max(H, psi(..))
  void output_results(const unsigned int cycle) const;

  // AMR
  void refine_mesh(unsigned int refine_cycle);

  // utilities
  inline double w_prime(const double phi) const { return use_AT1 ? 1.0 : 2.0 * phi; }
  inline double w_double(const double /*phi*/) const { return use_AT1 ? 0.0 : 2.0; }
  inline double g(const double phi) const { return (1.0 - phi) * (1.0 - phi) + kappa; }
  inline double g_prime(const double phi) const { return -2.0 * (1.0 - phi); }
  inline double g_double(const double /*phi*/) const { return 2.0; }

  // split (optional)
  double psi_plus(const SymmetricTensor<2,dim> &eps) const; // tension part
  double psi_full(const SymmetricTensor<2,dim> &eps) const; // full energy

private:
  // mesh & DoFs
  Triangulation<dim> triangulation;

  const unsigned int fe_u_order  = 2;
  const unsigned int fe_phi_order = 2;

  FESystem<dim>   fe_u;       // vector FE for displacement
  DoFHandler<dim> dof_handler_u;

  FE_Q<dim>       fe_phi;     // scalar FE for phi and H
  DoFHandler<dim> dof_handler_phi;

  AffineConstraints<double> constraints_u;
  AffineConstraints<double> constraints_phi;

  // linear algebra
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_jacobian;
  Vector<double>       system_residual;
  Vector<double>       system_delta;

  // state vectors
  Vector<double> solution_u;       // displacement
  Vector<double> solution_phi;     // phase field
  Vector<double> old_solution_phi; // for irreversibility clamp
  Vector<double> history_H;        // history field H

  // sizes
  unsigned int dofs_u_total = 0;
  unsigned int dofs_phi_total = 0;
  unsigned int system_size = 0;

  // material
  double E, nu, lambda, mu;
  double Gc, ell, kappa;
  bool   use_AT1;
  double cw;

  // loading
  unsigned int n_steps;
  double delta_max;

  // Newton
  unsigned int max_newton_iter;
  double tol_newton;

  // AMR
  unsigned int refine_every;
  unsigned int max_refine_levels;

  // options
  bool use_tension_split; // if true, use psi_plus in H

  TimerOutput timer;
};









// ----------------- IMPLEMENTATION -----------------






//SEGMENT - 1
template <int dim>
PhaseFieldMonolithicConsistentAMR<dim>::PhaseFieldMonolithicConsistentAMR()
  : fe_u(FE_Q<dim>(fe_u_order), dim)
  , dof_handler_u(triangulation)
  , fe_phi(fe_phi_order)
  , dof_handler_phi(triangulation)
  , E(210e3)
  , nu(0.31)
  , Gc(2.5)
  , ell(0.05)
  , kappa(1e-7)
  , use_AT1(true)
  , cw(use_AT1 ? 2.0/3.0 : 1.0/2.0)
  , n_steps(200)        // smaller load increments for robustness
  , delta_max(0.3)
  , max_newton_iter(50)
  , tol_newton(1e-8)
  , refine_every(2)
  , max_refine_levels(3)
  , use_tension_split(false)
  , timer(std::cout, TimerOutput::summary, TimerOutput::wall_times)
{
  lambda = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
  mu     = E / (2.0 * (1.0 + nu));
}






//SEGMENT - 2
template <int dim>
void PhaseFieldMonolithicConsistentAMR<dim>::make_grid()
{
  const double W = 2.0;
  const double H = 2.0;

  GridGenerator::subdivided_hyper_rectangle(triangulation,
                                            {50U, 50U},
                                            Point<dim>(0.0, 0.0),
                                            Point<dim>(W, H));

  for (auto &cell : triangulation.active_cell_iterators())
    for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
      if (cell->face(f)->at_boundary())
      {
        const auto c = cell->face(f)->center();
        if (std::fabs(c[0] - 0.0) < 1e-12) cell->face(f)->set_boundary_id(0); // left
        else if (std::fabs(c[0] - W) < 1e-12) cell->face(f)->set_boundary_id(1); // right
      }
}






//SEGMENT - 3
template <int dim>
void PhaseFieldMonolithicConsistentAMR<dim>::setup_system()
{
  dof_handler_u.distribute_dofs(fe_u);
  dof_handler_phi.distribute_dofs(fe_phi);

  dofs_u_total   = dof_handler_u.n_dofs();
  dofs_phi_total = dof_handler_phi.n_dofs();
  system_size    = dofs_u_total + dofs_phi_total;

  constraints_u.clear();
  VectorTools::interpolate_boundary_values(dof_handler_u,
                                           0,
                                           Functions::ZeroFunction<dim>(dim),
                                           constraints_u); // left fixed
  constraints_u.close();

  constraints_phi.clear(); // natural BC for phi
  constraints_phi.close();

  DynamicSparsityPattern dsp(system_size, system_size);

  std::vector<types::global_dof_index> loc_u(fe_u.n_dofs_per_cell());
  std::vector<types::global_dof_index> loc_p(fe_phi.n_dofs_per_cell());

  auto cu = dof_handler_u.begin_active();
  auto cp = dof_handler_phi.begin_active();
  for (; cu != dof_handler_u.end(); ++cu, ++cp)
  {
    cu->get_dof_indices(loc_u);
    cp->get_dof_indices(loc_p);

    for (auto i : loc_u) for (auto j : loc_u) dsp.add(i, j);
    for (auto i : loc_p) for (auto j : loc_p) dsp.add(dofs_u_total + i, dofs_u_total + j);

    for (auto iu : loc_u)
      for (auto jp : loc_p)
      {
        dsp.add(iu, dofs_u_total + jp);
        dsp.add(dofs_u_total + jp, iu);
      }
  }

  sparsity_pattern.copy_from(dsp);
  system_jacobian.reinit(sparsity_pattern);
  system_residual.reinit(system_size);
  system_delta.reinit(system_size);

  solution_u.reinit(dofs_u_total);
  solution_phi.reinit(dofs_phi_total);
  old_solution_phi.reinit(dofs_phi_total);
  history_H.reinit(dofs_phi_total);
}






//SEGMENT - 4
template <int dim>
void PhaseFieldMonolithicConsistentAMR<dim>::initialize_fields()
{
  // initial slit band (phi=1) around y ~ 2.5 and x < 2.0
  MappingQ1<dim> mapping;
  std::vector<Point<dim>> sp(dof_handler_phi.n_dofs());
  DoFTools::map_dofs_to_support_points(mapping, dof_handler_phi, sp);

  for (unsigned int i=0; i<sp.size(); ++i)
  {
    const auto &p = sp[i];
    // domain y in [0,2], place band near y = 1.0
    const bool in_band = (p[1] > 0.95) && (p[1] < 1.05) && (p[0] < 1.0);
    solution_phi[i]     = in_band ? 1.0 : 0.0;
    old_solution_phi[i] = solution_phi[i];
    history_H[i]        = 0.0;
  }
  solution_u = 0.0;
}






//SEGMENT - 5
template <int dim>
double PhaseFieldMonolithicConsistentAMR<dim>::psi_plus(
    const SymmetricTensor<2,dim> &eps) const
{
  const auto eig = dealii::eigenvalues(eps);
  double tr_pos = 0.0, pos_sq = 0.0;
  for (unsigned int i=0;i<dim;++i){ const double ep=std::max(0.0,eig[i]); tr_pos+=ep; pos_sq+=ep*ep; }
  return 0.5*lambda*tr_pos*tr_pos + mu*pos_sq;
}

template <int dim>
double PhaseFieldMonolithicConsistentAMR<dim>::psi_full(
    const SymmetricTensor<2,dim> &eps) const
{
  const double tr = trace(eps);
  return 0.5*lambda*tr*tr + mu*(eps*eps);
}






//SEGMENT - 6
template <int dim>
void PhaseFieldMonolithicConsistentAMR<dim>::assemble_residual_and_jacobian()
{
  system_jacobian = 0;
  system_residual = 0;

  const QGauss<dim> quad(std::max(fe_u.degree, fe_phi.degree)+1);
  FEValues<dim> fev_u(fe_u, quad, update_values|update_gradients|update_quadrature_points|update_JxW_values);
  FEValues<dim> fev_p(fe_phi, quad, update_values|update_gradients|update_quadrature_points|update_JxW_values);

  const FEValuesExtractors::Vector U(0);

  const unsigned int n_q = quad.size();
  const unsigned int nu  = fe_u.n_dofs_per_cell();
  const unsigned int np  = fe_phi.n_dofs_per_cell();

  FullMatrix<double> Kuu(nu,nu), Kup(nu,np), Kpu(np,nu), Kpp(np,np);
  Vector<double>     Ru(nu), Rp(np);

  std::vector<types::global_dof_index> lu(nu), lp(np);

  auto cu = dof_handler_u.begin_active();
  auto cp = dof_handler_phi.begin_active();
  for (; cu != dof_handler_u.end(); ++cu, ++cp)
  {
    fev_u.reinit(cu);
    fev_p.reinit(cp);

    Kuu = 0; Kup = 0; Kpu = 0; Kpp = 0;
    Ru  = 0; Rp  = 0;

    cu->get_dof_indices(lu);
    cp->get_dof_indices(lp);

    // gather nodal values on cell
    std::vector<double> phi_n(np), H_n(np);
    for (unsigned int i=0; i<np; ++i)
    {
      phi_n[i] = solution_phi[lp[i]];
      H_n[i]   = history_H[lp[i]];
    }
    std::vector<double> u_loc(nu);
    for (unsigned int i=0; i<nu; ++i) u_loc[i] = solution_u[lu[i]];

    for (unsigned int q=0; q<n_q; ++q)
    {
      // interpolate φ, H, ∇φ
      double phi_q = 0.0, Hq = 0.0;
      Tensor<1,dim> grad_phi_q;
      for (unsigned int i=0; i<np; ++i)
      {
        const double Ni = fev_p.shape_value(i,q);
        phi_q += phi_n[i]*Ni;
        Hq    += H_n[i]*Ni;
        grad_phi_q += phi_n[i]*fev_p.shape_grad(i,q);
      }

      // strain from u (symmetric gradient)
      SymmetricTensor<2,dim> eps_q;
      for (unsigned int i=0; i<nu; ++i)
        eps_q += fev_u[U].symmetric_gradient(i,q) * u_loc[i];

      const double gq        = g(phi_q);
      const double gprime_q  = g_prime(phi_q);
      const double gdouble_q = g_double(phi_q);
      const double wprime_q  = w_prime(phi_q);
      const double wdouble_q = w_double(phi_q);

      const double JxW = fev_u.JxW(q);
      const double factor = Gc / (2.0 * cw * ell);

      // precompute shape data
      std::vector<SymmetricTensor<2,dim>> sym_grad_u_i(nu);
      std::vector<double> trace_sym_i(nu);
      for (unsigned int i=0; i<nu; ++i)
      {
        sym_grad_u_i[i] = fev_u[U].symmetric_gradient(i,q);
        trace_sym_i[i]  = trace(sym_grad_u_i[i]);
      }
      const double tr_eps = trace(eps_q);

      // ------------ Kuu
      for (unsigned int i=0; i<nu; ++i)
        for (unsigned int j=0; j<nu; ++j)
        {
          const double eps_dot_ij = (sym_grad_u_i[i] * sym_grad_u_i[j]); // A:B
          const double val = gq*(lambda * trace_sym_i[i] * trace_sym_i[j] + 2.0 * mu * eps_dot_ij);
          Kuu(i,j) += val * JxW;
        }

      // ------------ Kup (∂Ru/∂φ)
      for (unsigned int i=0; i<nu; ++i)
      {
        const double eps_dot_i = (eps_q * sym_grad_u_i[i]);
        const double S_i = lambda * tr_eps * trace_sym_i[i] + 2.0 * mu * eps_dot_i;
        for (unsigned int j=0; j<np; ++j)
        {
          const double Nj = fev_p.shape_value(j,q);
          Kup(i,j) += gprime_q * Nj * S_i * JxW;
        }
      }

      // ------------ Kpp (grad + reaction)
      for (unsigned int i=0; i<np; ++i)
        for (unsigned int j=0; j<np; ++j)
        {
          const auto gradNi = fev_p.shape_grad(i,q);
          const auto gradNj = fev_p.shape_grad(j,q);
          Kpp(i,j) += (factor * ell * ell) * (gradNi * gradNj) * JxW;
        }

      for (unsigned int i=0; i<np; ++i)
      {
        const double Ni     = fev_p.shape_value(i,q);
        const auto   gradNi = fev_p.shape_grad(i,q);

        // residual for φ
        Rp(i) += (factor * ell * ell) * (grad_phi_q * gradNi) * JxW;                   // diffusion
        Rp(i) += ( gprime_q * Hq + factor * (wprime_q/2.0) ) * Ni * JxW;               // reaction

        // linearization (diagonal terms)
        for (unsigned int j=0; j<np; ++j)
        {
          const double Nj = fev_p.shape_value(j,q);
          Kpp(i,j) += (gdouble_q * Hq) * Ni * Nj * JxW;
          Kpp(i,j) += (factor * (wdouble_q/2.0)) * Ni * Nj * JxW;
        }
      }

      // ------------ Kpu (∂Rφ/∂u)
      for (unsigned int i=0; i<np; ++i)
      {
        const double Ni = fev_p.shape_value(i,q);
        for (unsigned int k=0; k<nu; ++k)
        {
          const auto &Bk = sym_grad_u_i[k];
          double dpsi_du_k = 0.0;
          dpsi_du_k += lambda * tr_eps * trace(Bk);
          dpsi_du_k += 2.0 * mu * (eps_q * Bk);
          Kpu(i,k) += gprime_q * dpsi_du_k * Ni * JxW;
        }
      }
    } // q

    // Ru = -Kuu*u_local  (internal forces only)
    for (unsigned int i=0; i<nu; ++i)
    {
      double tmp = 0.0;
      for (unsigned int j=0; j<nu; ++j)
        tmp += Kuu(i,j) * solution_u[lu[j]];
      Ru(i) -= tmp;
    }

    // Scatter to global system
    for (unsigned int i=0; i<nu; ++i)
    {
      const auto Gi = lu[i];
      for (unsigned int j=0; j<nu; ++j) system_jacobian.add(Gi, lu[j], Kuu(i,j));
      for (unsigned int j=0; j<np; ++j) system_jacobian.add(Gi, dofs_u_total + lp[j], Kup(i,j));
      system_residual[Gi] += Ru(i);
    }
    for (unsigned int i=0; i<np; ++i)
    {
      const auto Gi = dofs_u_total + lp[i];
      for (unsigned int j=0; j<nu; ++j) system_jacobian.add(Gi, lu[j], Kpu(i,j));
      for (unsigned int j=0; j<np; ++j) system_jacobian.add(Gi, dofs_u_total + lp[j], Kpp(i,j));
      system_residual[Gi] += Rp(i);
    }
  }

  // constraints
  constraints_u.condense(system_jacobian, system_residual);
  constraints_phi.condense(system_jacobian, system_residual);
}







//SEGMENT - 7
template <int dim>
void PhaseFieldMonolithicConsistentAMR<dim>::solve_linear_system()
{
  SparseDirectUMFPACK solver;
  solver.initialize(system_jacobian);
  Vector<double> rhs = system_residual;
  rhs *= -1.0;
  solver.vmult(system_delta, rhs);
  constraints_u.distribute(system_delta);
  constraints_phi.distribute(system_delta);
}






//SEGMENT - 8
template <int dim>
void PhaseFieldMonolithicConsistentAMR<dim>::update_solution_and_enforce_bounds(const double alpha)
{
  for (unsigned int i=0; i<dofs_u_total; ++i)
    solution_u[i] += alpha * system_delta[i];

  for (unsigned int i=0; i<dofs_phi_total; ++i)
  {
    const double val = solution_phi[i] + alpha * system_delta[dofs_u_total + i];
    const double clamped = std::min(1.0, std::max(0.0, val));
    solution_phi[i] = std::max(old_solution_phi[i], clamped); // irreversibility
  }
}






//SEGMENT - 9
template <int dim>
void PhaseFieldMonolithicConsistentAMR<dim>::update_history_field()
{
  const QGauss<dim> quad(std::max(fe_u.degree, fe_phi.degree)+1);
  FEValues<dim> fev_u(fe_u, quad, update_values|update_gradients|update_quadrature_points|update_JxW_values);
  FEValues<dim> fev_p(fe_phi, quad, update_values);

  const FEValuesExtractors::Vector U(0);

  Vector<double> lumped_mass(dofs_phi_total);
  Vector<double> rhs(dofs_phi_total);
  lumped_mass = 0.0; rhs = 0.0;

  std::vector<types::global_dof_index> lu(fe_u.n_dofs_per_cell()), lp(fe_phi.n_dofs_per_cell());

  auto cu = dof_handler_u.begin_active();
  auto cp = dof_handler_phi.begin_active();
  for (; cu != dof_handler_u.end(); ++cu, ++cp)
  {
    fev_u.reinit(cu);
    fev_p.reinit(cp);

    cu->get_dof_indices(lu);
    cp->get_dof_indices(lp);

    const unsigned int n_q = fev_u.n_quadrature_points;
    const unsigned int nu  = fe_u.n_dofs_per_cell();
    const unsigned int np  = fe_phi.n_dofs_per_cell();

    std::vector<double> u_loc(nu);
    for (unsigned int i=0; i<nu; ++i) u_loc[i] = solution_u[lu[i]];

    for (unsigned int q=0; q<n_q; ++q)
    {
      SymmetricTensor<2,dim> eps_q;
      for (unsigned int i=0; i<nu; ++i)
        eps_q += fev_u[U].symmetric_gradient(i,q) * u_loc[i];

      const double psi_q = use_tension_split ? psi_plus(eps_q) : psi_full(eps_q);
      const double JxW   = fev_u.JxW(q);

      for (unsigned int i=0; i<np; ++i)
      {
        const double Ni = fev_p.shape_value(i,q);
        lumped_mass[lp[i]] += Ni * JxW;
        rhs[lp[i]]         += Ni * psi_q * JxW;
      }
    }
  }

  for (unsigned int i=0; i<dofs_phi_total; ++i)
  {
    const double proj = (lumped_mass[i] > 0.0 ? rhs[i]/lumped_mass[i] : 0.0);
    history_H[i] = std::max(history_H[i], proj);
  }
}






//SEGMENT - 10
template <int dim>
void PhaseFieldMonolithicConsistentAMR<dim>::output_results(const unsigned int cycle) const
{
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler_u);
    std::vector<std::string> names(dim);
    for (unsigned int d=0; d<dim; ++d) names[d] = "u" + std::to_string(d);
    data_out.add_data_vector(solution_u, names);
    data_out.build_patches();
    std::ofstream f("solution_u-" + std::to_string(cycle) + ".vtu");
    data_out.write_vtu(f);
  }
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler_phi);
    data_out.add_data_vector(solution_phi, "phi");
    data_out.add_data_vector(history_H,   "H");
    data_out.build_patches();
    std::ofstream f("solution_phiH-" + std::to_string(cycle) + ".vtu");
    data_out.write_vtu(f);
  }
  std::ofstream pvd("results.pvd");
  pvd << "<?xml version=\"1.0\"?>\n<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
      << "  <Collection>\n";
  for (unsigned int c=0; c<=cycle; ++c)
  {
    pvd << "    <DataSet timestep=\"" << c << "\" group=\"\" part=\"0\" file=\"solution_u-" << c << ".vtu\"/>\n";
    pvd << "    <DataSet timestep=\"" << c << "\" group=\"\" part=\"0\" file=\"solution_phiH-" << c << ".vtu\"/>\n";
  }
  pvd << "  </Collection>\n</VTKFile>\n";
}






//SEGMENT - 11
template <int dim>
void PhaseFieldMonolithicConsistentAMR<dim>::refine_mesh(unsigned int /*refine_cycle*/)
{
  Vector<float> est(triangulation.n_active_cells());
  KellyErrorEstimator<dim>::estimate(dof_handler_phi,
                                     QGauss<dim-1>(fe_phi.degree + 1),
                                     std::map<types::boundary_id, const Function<dim> *>(),
                                     solution_phi,
                                     est);

  const double refine_fraction  = 0.20;
  const double coarsen_fraction = 0.05;
  GridRefinement::refine_and_coarsen_fixed_number(triangulation, est, refine_fraction, coarsen_fraction);

  // depth control
  int min_level = std::numeric_limits<int>::max();
  for (auto cell : triangulation.active_cell_iterators())
    min_level = std::min(min_level, static_cast<int>(cell->level()));
  const int max_allowed = min_level + static_cast<int>(max_refine_levels);
  for (auto cell : triangulation.active_cell_iterators())
  {
    if ((int)cell->level() >= max_allowed) cell->clear_refine_flag();
    if ((int)cell->level() <= min_level)   cell->clear_coarsen_flag();
  }

  // --- replace your refine_mesh() transfer block ---

  // BEFORE CC&R
  SolutionTransfer<dim, Vector<double>> tr_u(dof_handler_u);
  SolutionTransfer<dim, Vector<double>> tr_p(dof_handler_phi);

  tr_u.prepare_for_coarsening_and_refinement(solution_u);

  // prepare BOTH phi and H with a single transferor
  std::vector<const Vector<double>*> phiH_src = { &solution_phi, &history_H };
  tr_p.prepare_for_coarsening_and_refinement(phiH_src);

  triangulation.execute_coarsening_and_refinement();

  // DO NOT call distribute_dofs manually here; setup_system() does it safely
  setup_system();  // this redistributes DoFs and resizes vectors

  // interpolate back
  tr_u.interpolate(solution_u);

  std::vector<Vector<double>*> phiH_dst = { &solution_phi, &history_H };
  tr_p.interpolate(phiH_dst);

  // keep irreversibility
  old_solution_phi = solution_phi;

}






//SEGMENT - 12
template <int dim>
void PhaseFieldMonolithicConsistentAMR<dim>::run()
{
  make_grid();
  setup_system();
  initialize_fields();
  output_results(0);

  for (unsigned int step=1; step<=n_steps; ++step)
  {
    const double disp = (double)step * delta_max / (double)n_steps;

    // BCs: left fixed; right impose uy=disp, ux free
    class RightDisp : public Function<dim>
    {
    public:
      RightDisp(double v) : Function<dim>(dim), val(v) {}
      void vector_value(const Point<dim> &, Vector<double> &v) const override
      { v = 0.0; v[1] = val; }
      double val;
    } right_disp(disp);

    constraints_u.clear();
    VectorTools::interpolate_boundary_values(dof_handler_u, 0,
                                             Functions::ZeroFunction<dim>(dim),
                                             constraints_u);
    VectorTools::interpolate_boundary_values(dof_handler_u, 1,
                                             right_disp, constraints_u);
    constraints_u.close();

    // store φ_old; update H from current u
    old_solution_phi = solution_phi;
    update_history_field();

    // --- Newton ---
    bool converged = false;
    for (unsigned int it=0; it<max_newton_iter; ++it)
    {
      assemble_residual_and_jacobian();
      const double R0 = system_residual.l2_norm();
      std::cout << "Step " << step << " Newton " << it
                << " ||R|| = " << R0 << std::endl;

      if (R0 < tol_newton) { converged = true; break; }

      solve_linear_system();

      // relative increment
      double du_norm = 0.0, u_norm = 0.0, dp_norm = 0.0, p_norm = 0.0;
      for (unsigned int i=0;i<dofs_u_total;++i)
      { du_norm += system_delta[i]*system_delta[i]; u_norm += solution_u[i]*solution_u[i]; }
      for (unsigned int i=0;i<dofs_phi_total;++i)
      { const double d=system_delta[dofs_u_total+i]; dp_norm += d*d; p_norm += solution_phi[i]*solution_phi[i]; }
      const double rel_inc = std::sqrt(du_norm+dp_norm) / (1e-16 + std::sqrt(u_norm+p_norm));
      if (rel_inc < 1e-10) { converged = true; break; }

      // backtracking
      double alpha = 1.0, best_alpha = 0.0;
      double best_R = R0;
      const double rho = 0.5;
      const double c   = 1e-4;

      Vector<double> save_u = solution_u, save_p = solution_phi;
      bool decreased = false;
      for (unsigned int bt=0; bt<12; ++bt)
      {
        solution_u = save_u; solution_phi = save_p;
        update_solution_and_enforce_bounds(alpha);

        assemble_residual_and_jacobian();
        const double Rtrial = system_residual.l2_norm();

        if (Rtrial < best_R) { best_R = Rtrial; best_alpha = alpha; decreased = true; }
        if (Rtrial <= (1.0 - c*alpha) * R0) { decreased = true; best_alpha = alpha; break; }

        alpha *= rho;
      }

      if (!decreased)
      {
        solution_u = save_u; solution_phi = save_p;
        const double tiny = 1e-3;
        update_solution_and_enforce_bounds(tiny);
        assemble_residual_and_jacobian();
        const double Rtiny = system_residual.l2_norm();
        if (Rtiny >= R0)
        {
          std::cout << "Stagnation: reducing load increment\n";
          break;
        }
      }
      else
      {
        solution_u = save_u; solution_phi = save_p;
        update_solution_and_enforce_bounds(best_alpha);
      }
    } // <-- closes Newton loop

    if (!converged)
      std::cout << "Warning: Newton did not converge in step " << step << "\n";

    output_results(step);

    if ((step % refine_every == 0) && (step != n_steps))
    {
      std::cout << "Refining mesh after step " << step << std::endl;
      refine_mesh(step/refine_every);
      output_results(step);
    }
  } // <-- closes step loop
}   // <-- closes run()

///////////////////////////////////////////////////







int main()
{
  try
  {
    deallog.depth_console(0);
    PhaseFieldMonolithicConsistentAMR<2> app;
    app.run();
  }
  catch (std::exception &e)
  {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << "Unknown exception!" << std::endl;
    return 1;
  }
  return 0;
}
