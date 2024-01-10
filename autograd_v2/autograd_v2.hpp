#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <deque>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <sstream>
#include <stack>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

#if NDEBUG
#define CHECK_NAN(x)                                                           \
  do {                                                                         \
    auto d = x;                                                                \
    assert(!std::isnan(d));                                                    \
    return d;                                                                  \
  } while (0)
#else
#define CHECK_NAN(x) return x
#endif

namespace aks {

template <typename T> using vec_t = std::vector<T>;
template <typename T> using deq_t = std::deque<T>;
template <typename T, size_t N> using arr_t = std::array<T, N>;
template <typename T> using stack_t = std::stack<T, std::vector<T>>;
using real_t = double;
using reals_t = vec_t<real_t>;
using idx_t = size_t;
constexpr idx_t const sntnl_idx = std::numeric_limits<idx_t>::max();

#pragma region details

// namespace detail {
template <typename F, typename T, typename... Ts>
auto zipped_op(F &&f, T &&t, Ts &&...ts) {
  using ret_t = decltype(f(t.front(), ts.front()...));
  std::vector<ret_t> ret;
  ret.reserve(t.size());
  for (size_t i = 0; i < t.size(); ++i) {
    ret.emplace_back(f(t[i], ts[i]...));
  }
  return ret;
}

//} // namespace detail

struct tape_t;

struct node;

struct var;

struct var {
  var(tape_t *t, node *n) : t_(t), n_(n) {}
  var() : t_(nullptr), n_(nullptr) {}

  var(var const &r) = default;
  var(var &&r) = default;

  var(real_t r, tape_t *t);

  var clone() const { return var{value(), t_}; }

  real_t value() const;

  tape_t &t() { return *t_; }
  tape_t &t() const { return *t_; }
  node &n() { return *n_; }
  node const &n() const { return *n_; }
  node *np() const { return n_; }

  var &operator=(var const &r) = default;
  var &operator=(var &&r) = default;

  var &operator+=(var const &r);
  var &operator-=(var const &r);
  var &operator*=(var const &r);
  var &operator/=(var const &r);
  var operator-() const;

  // var& operator/=(var const& r);

  bool is_alive() const { return t_ != nullptr && n_ != nullptr; }

private:
  mutable tape_t *t_ = nullptr;
  node *n_ = nullptr;
};

typedef void (*back_f_t)(tape_t *, node *);

struct back_f {
  const char *n_ = nullptr;
  back_f_t f_ = nullptr;
};

struct node {
  real_t v_ = real_t{};
  back_f backwards_;
  vec_t<node *> ls_, rs_;
  idx_t idx_ = sntnl_idx;
  bool is_leaf() const { return ls_.empty() && rs_.empty(); }
};

struct tape_t {

  node *new_node() {
    auto ret = &(nodes_.emplace_back());
    ret->idx_ = nodes_.size() - 1;
    return ret;
  }

  var new_variable(real_t const &r) {
    node *n = new_node();
    n->v_ = r;
    return {this, n};
  }

  void fill_grads() { grads_.resize(nodes_.size()); }

  void zero_grad() { grads_.clear(); }

  void keep_only(std::tuple<idx_t, idx_t> const &ds) {
    nodes_.resize(std::get<0>(ds));
    grads_.resize(std::get<1>(ds));
  }

  void reset() {
    keep_only({0, 0});
    stack_t<std::tuple<idx_t, idx_t>> clear;
    saved_state_.swap(clear);
  }
  void push_state() { saved_state_.push({nodes_.size(), grads_.size()}); }

  void pop_state() {
    if (!saved_state_.empty()) {
      auto to_restore = saved_state_.top();
      saved_state_.pop();
      keep_only(to_restore);
    } else {
      reset();
    }
  }

  deq_t<node> nodes_;
  vec_t<var> grads_;
  stack_t<std::tuple<idx_t, idx_t>> saved_state_;
};

auto &grad(var &n) { return n.t().grads_[n.n().idx_]; }
auto const &grad(var const &n) { return n.t().grads_[n.n().idx_]; }

#pragma endregion

#pragma region mixes

struct exp_mix {
  static real_t apply(real_t a) { CHECK_NAN(std::exp(a)); }
  static back_f bf() { return {"exp", backward}; }
  static void backward(tape_t *t, node *n);
};

struct log_mix {
  static real_t apply(real_t a) { CHECK_NAN(std::log(a)); }
  static back_f bf() { return {"log", backward}; }
  static void backward(tape_t *t, node *n);
};

struct tanh_mix {
  static real_t apply(real_t a) { return std::tanh(a); }
  static back_f bf() { return {"tanh", backward}; }
  static void backward(tape_t *t, node *n);
};

struct neg_mix {
  static real_t apply(real_t a) { return -a; }
  static back_f bf() { return {"neg", backward}; }
  static void backward(tape_t *t, node *n);
};

struct sin_mix {
  static real_t apply(real_t a) { CHECK_NAN(std::sin(a)); }
  static back_f bf() { return {"sin", backward}; }
  static void backward(tape_t *t, node *n);
};

struct cos_mix {
  static real_t apply(real_t a) { CHECK_NAN(std::cos(a)); }
  static back_f bf() { return {"cos", backward}; }
  static void backward(tape_t *t, node *n);
};

struct sqrt_mix {
  static real_t apply(real_t a) { CHECK_NAN(std::sqrt(a)); }
  static back_f bf() { return {"sqrt", backward}; }
  static void backward(tape_t *t, node *n);
};

struct relu_mix {
  static real_t apply(real_t a) { return (a > 0.0) ? a : 0.0; }
  static back_f bf() { return {"relu", backward}; }
  static void backward(tape_t *t, node *n);
};

// struct id_mix {
//   static real_t apply(real_t a) { return a; }
//   static back_f bf() { return {"id", backward}; }
//   static void backward(tape_t *t, node *n);
// };

struct add_mix {
  static real_t apply(real_t a, real_t b) { return a + b; }
  static back_f bf() { return {"add", backward}; }
  static void backward(tape_t *t, node *n);
};

struct sub_mix {
  static real_t apply(real_t a, real_t b) { return a - b; }
  static back_f bf() { return {"sub", backward}; }
  static void backward(tape_t *t, node *n);
};

struct mul_mix {
  static real_t apply(real_t a, real_t b) { return a * b; }
  static back_f bf() { return {"mul", backward}; }
  static void backward(tape_t *t, node *n);
};

struct div_mix {
  static real_t apply(real_t a, real_t b) { return a / b; }
  static back_f bf() { return {"div", backward}; }
  static void backward(tape_t *t, node *n);
};

struct pow_mix {
  static real_t apply(real_t a, real_t b) { return std::pow(a, b); }
  static back_f bf() { return {"pow", backward}; }
  static void backward(tape_t *t, node *n);
};

template <typename mix> struct u_op_mix : mix {
  using mix::apply;
  using mix::bf;
  static real_t fwd(real_t const &a) { return apply(a); }
  static var fwd(var const &a) {
    node *new_node = a.t().new_node();
    new_node->v_ = fwd(a.n().v_);
    new_node->backwards_ = bf();
    new_node->ls_.push_back(a.np());
    return var{&a.t(), new_node};
  }
};

template <typename mix> struct op_mix : mix {
  using mix::apply;
  using mix::bf;
  static real_t fwd(real_t const &a, real_t const &b) { return apply(a, b); }
  static var fwd(var const &a, var const &b) {
    node *new_node = a.t().new_node();
    new_node->v_ = fwd(a.n().v_, b.n().v_);
    new_node->backwards_ = bf();
    new_node->ls_.push_back(a.np());
    new_node->rs_.push_back(b.np());
    return var{&a.t(), new_node};
  }
};

struct dot_mix {
  static real_t start() { return 0.0; }
  static void apply(real_t &r, real_t a, real_t b) { r += (a * b); }
  static back_f bf() { return {"dot", backward}; }
  static void backward(tape_t *t, node *n);
};

struct asum_mix {
  static real_t start() { return 0.0; }
  static void apply(real_t &r, real_t a) { r += a; }
  static back_f bf() { return {"asum", backward}; }
  static void backward(tape_t *t, node *n);
};

struct gsum_mix {
  static real_t start() { return 1.0; }
  static void apply(real_t &r, real_t a) { r *= a; }
  static back_f bf() { return {"gsum", nullptr}; }
  static void backward(tape_t *t, node *n);
};

// struct max_mix {
//   static real_t start() { return -INFINITY; }
//   static void apply(real_t &r, real_t a) { r = std::max(r, a); }
//   static back_f bf() { return {"max", nullptr}; }
//   // static void backward(tape_t* t, node* n);
// };
//
// struct min_mix {
//   static real_t start() { return INFINITY; }
//   static void apply(real_t &r, real_t a) { r = std::min(r, a); }
//   static back_f bf() { return {"min", nullptr}; }
//   // static void backward(tape_t* t, node* n);
// };

template <typename mix> struct v_u_to_1_op_mix : mix {
  using mix::apply;
  using mix::bf;
  using mix::start;
  static real_t fwd_impl(vec_t<var> const &a, vec_t<node *> &ls) {
    real_t ret = start();
    for (int i = 0; i < a.size(); i++) {
      apply(ret, a[i].n().v_);
      ls.push_back(a[i].np());
    }
    return ret;
  }
  static var fwd(vec_t<var> const &a) {
    assert(a.size());
    node *new_node = a.front().t().new_node();
    new_node->v_ = fwd_impl(a, new_node->ls_);
    new_node->backwards_ = bf();
    return var{&a.front().t(), new_node};
  }
};

template <typename mix> struct v_b_to_1_op_mix : mix {
  using mix::apply;
  using mix::bf;
  using mix::start;
  static real_t fwd_impl(vec_t<var> const &a, vec_t<var> const &b,
                         vec_t<node *> &ls, vec_t<node *> &rs) {
    real_t ret = start();
    for (int i = 0; i < a.size(); i++) {
      apply(ret, a[i].n().v_, b[i].n().v_);
      ls.push_back(a[i].np());
      rs.push_back(b[i].np());
    }
    return ret;
  }
  static var fwd(vec_t<var> const &a, vec_t<var> const &b) {
    assert(a.size());
    assert(a.size() == b.size());
    node *new_node = a.front().t().new_node();
    new_node->v_ = fwd_impl(a, b, new_node->ls_, new_node->rs_);
    new_node->backwards_ = bf();
    return var{&a.front().t(), new_node};
  }
};

#pragma endregion

#pragma region operator_overloads

// auto identity(var a) { return u_op_mix<id_mix>::fwd(a); }

auto relu(var a) { return u_op_mix<relu_mix>::fwd(a); }
auto sqrt(var a) { return u_op_mix<sqrt_mix>::fwd(a); }

auto sin(var a) { return u_op_mix<sin_mix>::fwd(a); }
auto cos(var a) { return u_op_mix<cos_mix>::fwd(a); }
auto tanh(var a) { return u_op_mix<tanh_mix>::fwd(a); }
auto neg(var a) { return u_op_mix<neg_mix>::fwd(a); }

auto exp(var a) { return u_op_mix<exp_mix>::fwd(a); }
auto log(var a) { return u_op_mix<log_mix>::fwd(a); }

auto pow(var a, var b) { return op_mix<pow_mix>::fwd(a, b); }
auto operator+(var a, var b) { return op_mix<add_mix>::fwd(a, b); }
auto operator-(var a, var b) { return op_mix<sub_mix>::fwd(a, b); }
auto operator*(var a, var b) { return op_mix<mul_mix>::fwd(a, b); }
auto operator/(var a, var b) { return op_mix<div_mix>::fwd(a, b); }
auto operator^(var a, var b) { return pow(a, b); }

auto pow(var a, real_t b) { return pow(a, var{b, &a.t()}); }
auto operator+(var a, real_t b) { return a + var{b, &a.t()}; }
auto operator-(var a, real_t b) { return a - var{b, &a.t()}; }
auto operator*(var a, real_t b) { return a * var{b, &a.t()}; }
auto operator/(var a, real_t b) { return a / var{b, &a.t()}; }
auto operator^(var a, real_t b) { return pow(a, var{b, &a.t()}); }

auto pow(real_t a, var b) { return pow(var{a, &b.t()}, b); }
auto operator+(real_t a, var b) { return var{a, &b.t()} + b; }
auto operator-(real_t a, var b) { return var{a, &b.t()} - b; }
auto operator*(real_t a, var b) { return var{a, &b.t()} * b; }
auto operator/(real_t a, var b) { return var{a, &b.t()} / b; }
auto operator^(real_t a, var b) { return pow(var{a, &b.t()}, b); }

var::var(real_t r, tape_t *t) {
  t_ = t;
  n_ = t_->new_variable(r).np();
}

real_t var::value() const { return n().v_; }

var &var::operator+=(var const &r) {
  (*this) = (*this) + r;
  return *this;
}

var &var::operator-=(var const &r) {
  (*this) = (*this) - r;
  return *this;
}

var &var::operator*=(var const &r) {
  (*this) = (*this) * r;
  return *this;
}

var &var::operator/=(var const &r) {
  (*this) = (*this) / r;
  return *this;
}

var var::operator-() const { return *this * var(-1.0, t_); }

var dot(vec_t<var> const &a, vec_t<var> const &b) {
  return v_b_to_1_op_mix<dot_mix>::fwd(a, b);
}

var asum(vec_t<var> const &a) { return v_u_to_1_op_mix<asum_mix>::fwd(a); }
var gsum(vec_t<var> const &a) { return v_u_to_1_op_mix<gsum_mix>::fwd(a); }

var mean(vec_t<var> const &a) { return asum(a) / real_t(a.size()); }
var gmean(vec_t<var> const &a) { return gsum(a) / real_t(a.size()); }

var max(vec_t<var> const &a) {
  assert(a.size());
  var m = a[0];
  for (size_t i = 1; i < a.size(); ++i) {
    if (a[i].value() > m.value()) {
      m = a[i];
    }
  }
  return m;
}

var min(vec_t<var> const &a) {
  assert(a.size());
  var m = a[0];
  for (size_t i = 1; i < a.size(); ++i) {
    if (a[i].value() < m.value()) {
      m = a[i];
    }
  }
  return m;
}

#pragma endregion

#pragma region backward

template <typename F> void backward_grad_accumulate(var &x, F df) {
  if (grad(x).is_alive()) {
    grad(x) += df();
  } else {
    grad(x) = df();
  }
}

void backward_set_grad_if_not_alive(var &x) {
  if (!grad(x).is_alive()) {
    grad(x) = x.t().new_variable(0.0);
  }
}

void sqrt_mix::backward(tape_t *t, node *n) {
  var f{t, n};
  var &fg = grad(f);
  var l{t, n->ls_.front()};

  auto df = [&]() { return 0.5 * fg / sqrt(l); };

  backward_grad_accumulate(l, df);
}

void exp_mix::backward(tape_t *t, node *n) {
  var f{t, n};
  var &fg = grad(f);
  var l{t, n->ls_.front()};

  auto df = [&]() { return f * fg; };

  backward_grad_accumulate(l, df);
}

void log_mix::backward(tape_t *t, node *n) {
  var f{t, n};
  var &fg = grad(f);
  var l{t, n->ls_.front()};

  auto df = [&]() { return fg / l; };

  backward_grad_accumulate(l, df);
}

void neg_mix::backward(tape_t *t, node *n) {
  var f{t, n};
  var &fg = grad(f);
  var l{t, n->ls_.front()};

  auto df = [&]() { return -fg; };

  backward_grad_accumulate(l, df);
}

void sin_mix::backward(tape_t *t, node *n) {

  var f{t, n};
  var &fg = grad(f);
  var l{t, n->ls_.front()};

  auto df = [&]() { return cos(l) * fg; };

  backward_grad_accumulate(l, df);
}

void cos_mix::backward(tape_t *t, node *n) {
  var f{t, n};
  var &fg = grad(f);
  var l{t, n->ls_.front()};

  auto df = [&]() { return -sin(l) * fg; };

  backward_grad_accumulate(l, df);
}

void tanh_mix::backward(tape_t *t, node *n) {

  var f{t, n};
  var &fg = grad(f);
  var l{t, n->ls_.front()};

  auto df = [&]() { return (1.0 - (f ^ 2.0)) * fg; };

  backward_grad_accumulate(l, df);
}

void relu_mix::backward(tape_t *t, node *n) {

  var f{t, n};
  var &fg = grad(f);
  var l{t, n->ls_.front()};

  auto df = [&]() { return (l.value() > 0.0) * fg; };

  backward_grad_accumulate(l, df);
}

// void id_mix::backward(tape_t *t, node *n) {
//   var f{t, n};
//   var &fg = grad(f);
//   var l{t, n->ls_.front()};
//
//   var one = t->new_variable(1.0);
//
//   if (!grad(l).is_alive()) {
//     grad(l) = var(0.0, t);
//   }
//
//   grad(l) += grad(l) + fg;
// }

void dot_mix::backward(tape_t *t, node *n) {
  var f{t, n};
  var &fg = grad(f);

  for (size_t i = 0; i < n->rs_.size(); ++i) {
    var l{t, n->ls_[i]};
    var r{t, n->rs_[i]};

    auto dfdl = [&]() { return r * fg; };
    auto dfdr = [&]() { return l * fg; };

    backward_grad_accumulate(l, dfdl);
    backward_grad_accumulate(r, dfdr);
  }
}

void asum_mix::backward(tape_t *t, node *n) {
  var f{t, n};
  var &fg = grad(f);

  for (auto &ls : n->ls_) {
    var l{t, ls};
    backward_set_grad_if_not_alive(l);
    grad(l) += fg;
  }
}

void add_mix::backward(tape_t *t, node *n) {
  var f{t, n};
  var &fg = grad(f);
  var l{t, n->ls_.front()};
  var r{t, n->rs_.front()};

  backward_set_grad_if_not_alive(l);
  backward_set_grad_if_not_alive(r);

  grad(l) += fg;
  grad(r) += fg;
}

void sub_mix::backward(tape_t *t, node *n) {
  var f{t, n};
  var &fg = grad(f);
  var l{t, n->ls_.front()};
  var r{t, n->rs_.front()};

  backward_set_grad_if_not_alive(l);
  backward_set_grad_if_not_alive(r);

  grad(l) += fg;
  grad(r) -= fg;
}

void mul_mix::backward(tape_t *t, node *n) {
  var f{t, n};
  var &fg = grad(f);
  var l{t, n->ls_.front()};
  var r{t, n->rs_.front()};

  auto dfdl = [&]() { return r * fg; };
  auto dfdr = [&]() { return l * fg; };

  backward_grad_accumulate(l, dfdl);
  backward_grad_accumulate(r, dfdr);
}

void div_mix::backward(tape_t *t, node *n) {
  var f{t, n};
  var &fg = grad(f);
  var l{t, n->ls_.front()};
  var r{t, n->rs_.front()};

  auto dfdl = [&]() { return fg / r; };
  auto dfdr = [&]() { return fg * l / (r * r); };

  backward_grad_accumulate(l, dfdl);

  if (grad(r).is_alive())
    grad(r) -= dfdr(); // negative sign
  else
    grad(r) = -dfdr();
}

void pow_mix::backward(tape_t *t, node *n) {
  var f{t, n};
  var &fg = grad(f);
  var l{t, n->ls_.front()};
  var r{t, n->rs_.front()};

  auto dfdl = [&]() { return fg * (r * pow(l, r - 1.0)); };
  auto dfdr = [&]() { return fg * f * log(l); };

  backward_grad_accumulate(l, dfdl);
  backward_grad_accumulate(r, dfdr);
}

void backward_impl(tape_t *t, node *n) {
  if (n->backwards_.f_ != nullptr) {
    n->backwards_.f_(t, n);
  }
}

void backward(var v) {

  v.t().fill_grads();

  if (!grad(v).is_alive()) {
    grad(v) = v.t().new_variable(1.0);
  }

  for (size_t idx = v.n().idx_ + 1; idx != 0; --idx) {
    node *n = &(v.t().nodes_[idx - 1]);
    var n_ = {&v.t(), n};
    if (!grad(n_).is_alive()) {
      grad(n_) = v.t().new_variable(0.0);
    }
    backward_impl(&n_.t(), &n_.n());
  }
}

#pragma endregion

} // namespace aks

namespace aks {

std::ostream &operator<<(std::ostream &o, var const &vs) {
  o << std::setprecision(15) << "var(" << vs.value() << ";"
    << (vs.n().backwards_.n_ ? vs.n().backwards_.n_ : "null") << ";"
    << vs.t().nodes_.size() << ")";
  return o;
}

template <typename T>
std::ostream &operator<<(std::ostream &o, vec_t<T> const &vs) {
  o << "[";
  bool comma = false;
  for (auto const &v : vs) {
    if (comma) {
      o << ",";
    }
    o << v;
    comma = true;
  }
  o << "]";
  return o;
}

std::string to_string(node const &v) {
  std::stringstream ss;
  ss << v.v_;
  return ss.str();
}

inline std::string as_dot(var iv, size_t i, size_t cnt, size_t start_cnt) {
  auto check = [](auto x) { return x != nullptr; };

  std::stringstream ss;

  size_t idx = i + start_cnt;
  node &nd = iv.n();

  auto is_binary_op = [](std::string const &x) {
    return x == "add" || x == "sub" || x == "mul" || x == "div" || x == "pow";
  };
  auto is_unary_op = [](std::string const &x) {
    return x == "neg" || x == "tanh" || x == "relu" || x == "sin" ||
           x == "cos" || x == "identity" || x == "sqrt" || x == "log" ||
           x == "exp";
  };

  if (!nd.is_leaf()) {
    ss << "\n"
       << idx << " [label = \"[" << idx << "] " << std::setprecision(15)
       << to_string(nd) << "\", fontcolor=blue, color=blue];\n";
    if (is_binary_op(nd.backwards_.n_)) {
      if (check(nd.ls_.front())) {
        ss << nd.ls_.front()->idx_ + start_cnt << "->" << nd.backwards_.n_
           << cnt << "; ";
      }
      if (check(nd.rs_.front())) {
        ss << nd.rs_.front()->idx_ + start_cnt << "->" << nd.backwards_.n_
           << cnt << "; ";
      }
      if (check(nd.ls_.front()) || check(nd.rs_.front())) {
        ss << nd.backwards_.n_ << cnt << "->" << idx << "; ";
      }
    } else if (is_unary_op(nd.backwards_.n_)) {
      if (check(nd.ls_.front())) {
        ss << nd.ls_.front()->idx_ + start_cnt << "->" << nd.backwards_.n_
           << cnt << "; ";
      }
      if (check(nd.ls_.front())) {
        ss << nd.backwards_.n_ << cnt << "->" << idx << "; ";
      }
    }
  } else {
    ss << "\n"
       << idx << " [label = \"[" << idx << "] " << std::setprecision(15)
       << to_string(nd) << "\", fontcolor=darkgreen, color=darkgreen];\n";
  }

  return ss.str();
}

inline std::string as_dot(tape_t &t, size_t start_count = 0) {
  auto &data = t.nodes_;

  auto to_symbol = [](std::string const &x) -> std::string {
    if (x == "add")
      return "+";
    else if (x == "sub")
      return "-";
    else if (x == "mul")
      return "*";
    else if (x == "div")
      return "/";
    else
      return x;
  };

  auto to_preamble = [&](std::unordered_map<std::string, size_t> const &d) {
    std::stringstream ss;
    for (auto const &[o, c] : d) {
      if (!o.empty()) {
        ss << "{node[label = \"" << to_symbol(o)
           << "\", shape = circle, fontcolor=red, color=red] ";
        for (size_t i = start_count; i < c; ++i) {
          ss << o << i << "; ";
        }
        // add0; add1; };
        ss << "};\n";
      }
    }
    return ss.str();
  };

  std::unordered_map<std::string, size_t> opcnt;
  std::vector<std::string> strs;
  strs.reserve(data.size());
  for (size_t idx = 0; idx != data.size(); ++idx) {
    const node &d = data.at(idx);
    const char *o = d.backwards_.n_;
    std::string op = data[idx].backwards_.n_ ? data[idx].backwards_.n_ : "";
    // std::string op;
    if (!opcnt.contains(op)) {
      opcnt[op] = start_count;
    }
    size_t cnt = opcnt[op]++;
    std::string str = as_dot(var{&t, &data[idx]}, idx, cnt, start_count);
    strs.emplace_back(std::move(str));
  }
  std::stringstream ss;
  ss << "digraph G{\n";
  ss << "node[shape = record];\n";
  ss << to_preamble(opcnt);

  for (const auto &s : strs) {
    ss << s;
  }
  ss << "}\n";
  return ss.str();
}
} // namespace aks
