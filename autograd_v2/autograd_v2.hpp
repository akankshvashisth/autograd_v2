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
using re_t = double;
using vec_re_t = vec_t<re_t>;
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

struct var_t;

struct var_t {
  var_t(tape_t *t, node *n) : t_(t), n_(n) {}
  var_t() : t_(nullptr), n_(nullptr) {}

  var_t(var_t const &r) = default;
  var_t(var_t &&r) = default;

  var_t(re_t r, tape_t *t);

  var_t clone() const { return var_t{value(), t_}; }

  re_t value() const;

  tape_t &t() { return *t_; }
  tape_t &t() const { return *t_; }
  node &n() { return *n_; }
  node const &n() const { return *n_; }
  node *np() const { return n_; }

  var_t &operator=(var_t const &r) = default;
  var_t &operator=(var_t &&r) = default;

  var_t &operator+=(var_t const &r);
  var_t &operator-=(var_t const &r);
  var_t &operator*=(var_t const &r);
  var_t &operator/=(var_t const &r);
  var_t operator-() const;

  // var_t& operator/=(var_t const& r);

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
  re_t v_ = re_t{};
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

  var_t new_variable(re_t const &r) {
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
  vec_t<var_t> grads_;
  stack_t<std::tuple<idx_t, idx_t>> saved_state_;
};

auto &grad(var_t &n) { return n.t().grads_[n.n().idx_]; }
auto const &grad(var_t const &n) { return n.t().grads_[n.n().idx_]; }

#pragma endregion

#pragma region mixes

struct exp_mix {
  static re_t apply(re_t a) { CHECK_NAN(std::exp(a)); }
  static back_f bf() { return {"exp", backward}; }
  static void backward(tape_t *t, node *n);
};

struct log_mix {
  static re_t apply(re_t a) { CHECK_NAN(std::log(a)); }
  static back_f bf() { return {"log", backward}; }
  static void backward(tape_t *t, node *n);
};

struct tanh_mix {
  static re_t apply(re_t a) { return std::tanh(a); }
  static back_f bf() { return {"tanh", backward}; }
  static void backward(tape_t *t, node *n);
};

struct neg_mix {
  static re_t apply(re_t a) { return -a; }
  static back_f bf() { return {"neg", backward}; }
  static void backward(tape_t *t, node *n);
};

struct sin_mix {
  static re_t apply(re_t a) { CHECK_NAN(std::sin(a)); }
  static back_f bf() { return {"sin", backward}; }
  static void backward(tape_t *t, node *n);
};

struct cos_mix {
  static re_t apply(re_t a) { CHECK_NAN(std::cos(a)); }
  static back_f bf() { return {"cos", backward}; }
  static void backward(tape_t *t, node *n);
};

struct sqrt_mix {
  static re_t apply(re_t a) { CHECK_NAN(std::sqrt(a)); }
  static back_f bf() { return {"sqrt", backward}; }
  static void backward(tape_t *t, node *n);
};

struct relu_mix {
  static re_t apply(re_t a) { return (a > 0.0) ? a : 0.0; }
  static back_f bf() { return {"relu", backward}; }
  static void backward(tape_t *t, node *n);
};

// struct id_mix {
//   static re_t apply(re_t a) { return a; }
//   static back_f bf() { return {"id", backward}; }
//   static void backward(tape_t *t, node *n);
// };

struct add_mix {
  static re_t apply(re_t a, re_t b) { return a + b; }
  static back_f bf() { return {"add", backward}; }
  static void backward(tape_t *t, node *n);
};

struct sub_mix {
  static re_t apply(re_t a, re_t b) { return a - b; }
  static back_f bf() { return {"sub", backward}; }
  static void backward(tape_t *t, node *n);
};

struct mul_mix {
  static re_t apply(re_t a, re_t b) { return a * b; }
  static back_f bf() { return {"mul", backward}; }
  static void backward(tape_t *t, node *n);
};

struct div_mix {
  static re_t apply(re_t a, re_t b) { return a / b; }
  static back_f bf() { return {"div", backward}; }
  static void backward(tape_t *t, node *n);
};

struct pow_mix {
  static re_t apply(re_t a, re_t b) { return std::pow(a, b); }
  static back_f bf() { return {"pow", backward}; }
  static void backward(tape_t *t, node *n);
};

template <typename mix> struct u_op_mix : mix {
  using mix::apply;
  using mix::bf;
  static re_t fwd(re_t const &a) { return apply(a); }
  static var_t fwd(var_t const &a) {
    node *new_node = a.t().new_node();
    new_node->v_ = fwd(a.n().v_);
    new_node->backwards_ = bf();
    new_node->ls_.push_back(a.np());
    return var_t{&a.t(), new_node};
  }
};

template <typename mix> struct op_mix : mix {
  using mix::apply;
  using mix::bf;
  static re_t fwd(re_t const &a, re_t const &b) { return apply(a, b); }
  static var_t fwd(var_t const &a, var_t const &b) {
    node *new_node = a.t().new_node();
    new_node->v_ = fwd(a.n().v_, b.n().v_);
    new_node->backwards_ = bf();
    new_node->ls_.push_back(a.np());
    new_node->rs_.push_back(b.np());
    return var_t{&a.t(), new_node};
  }
};

struct dot_mix {
  static re_t start() { return 0.0; }
  static void apply(re_t &r, re_t a, re_t b) { r += (a * b); }
  static back_f bf() { return {"dot", backward}; }
  static void backward(tape_t *t, node *n);
};

struct asum_mix {
  static re_t start() { return 0.0; }
  static void apply(re_t &r, re_t a) { r += a; }
  static back_f bf() { return {"asum", backward}; }
  static void backward(tape_t *t, node *n);
};

struct gsum_mix {
  static re_t start() { return 1.0; }
  static void apply(re_t &r, re_t a) { r *= a; }
  static back_f bf() { return {"gsum", nullptr}; }
  static void backward(tape_t *t, node *n);
};

// struct max_mix {
//   static re_t start() { return -INFINITY; }
//   static void apply(re_t &r, re_t a) { r = std::max(r, a); }
//   static back_f bf() { return {"max", nullptr}; }
//   // static void backward(tape_t* t, node* n);
// };
//
// struct min_mix {
//   static re_t start() { return INFINITY; }
//   static void apply(re_t &r, re_t a) { r = std::min(r, a); }
//   static back_f bf() { return {"min", nullptr}; }
//   // static void backward(tape_t* t, node* n);
// };

template <typename mix> struct v_u_to_1_op_mix : mix {
  using mix::apply;
  using mix::bf;
  using mix::start;
  static re_t fwd_impl(vec_t<var_t> const &a, vec_t<node *> &ls) {
    re_t ret = start();
    for (int i = 0; i < a.size(); i++) {
      apply(ret, a[i].n().v_);
      ls.push_back(a[i].np());
    }
    return ret;
  }
  static var_t fwd(vec_t<var_t> const &a) {
    assert(a.size());
    node *new_node = a.front().t().new_node();
    new_node->v_ = fwd_impl(a, new_node->ls_);
    new_node->backwards_ = bf();
    return var_t{&a.front().t(), new_node};
  }
};

template <typename mix> struct v_b_to_1_op_mix : mix {
  using mix::apply;
  using mix::bf;
  using mix::start;
  static re_t fwd_impl(vec_t<var_t> const &a, vec_t<var_t> const &b,
                       vec_t<node *> &ls, vec_t<node *> &rs) {
    re_t ret = start();
    for (int i = 0; i < a.size(); i++) {
      apply(ret, a[i].n().v_, b[i].n().v_);
      ls.push_back(a[i].np());
      rs.push_back(b[i].np());
    }
    return ret;
  }
  static var_t fwd(vec_t<var_t> const &a, vec_t<var_t> const &b) {
    assert(a.size());
    assert(a.size() == b.size());
    node *new_node = a.front().t().new_node();
    new_node->v_ = fwd_impl(a, b, new_node->ls_, new_node->rs_);
    new_node->backwards_ = bf();
    return var_t{&a.front().t(), new_node};
  }
};

#pragma endregion

#pragma region operator_overloads

// auto identity(var_t a) { return u_op_mix<id_mix>::fwd(a); }

auto relu(var_t a) { return u_op_mix<relu_mix>::fwd(a); }
auto sqrt(var_t a) { return u_op_mix<sqrt_mix>::fwd(a); }

auto sin(var_t a) { return u_op_mix<sin_mix>::fwd(a); }
auto cos(var_t a) { return u_op_mix<cos_mix>::fwd(a); }
auto tanh(var_t a) { return u_op_mix<tanh_mix>::fwd(a); }
auto neg(var_t a) { return u_op_mix<neg_mix>::fwd(a); }

auto exp(var_t a) { return u_op_mix<exp_mix>::fwd(a); }
auto log(var_t a) { return u_op_mix<log_mix>::fwd(a); }

auto pow(var_t a, var_t b) { return op_mix<pow_mix>::fwd(a, b); }
auto operator+(var_t a, var_t b) { return op_mix<add_mix>::fwd(a, b); }
auto operator-(var_t a, var_t b) { return op_mix<sub_mix>::fwd(a, b); }
auto operator*(var_t a, var_t b) { return op_mix<mul_mix>::fwd(a, b); }
auto operator/(var_t a, var_t b) { return op_mix<div_mix>::fwd(a, b); }
auto operator^(var_t a, var_t b) { return pow(a, b); }

auto pow(var_t a, re_t b) { return pow(a, var_t{b, &a.t()}); }
auto operator+(var_t a, re_t b) { return a + var_t{b, &a.t()}; }
auto operator-(var_t a, re_t b) { return a - var_t{b, &a.t()}; }
auto operator*(var_t a, re_t b) { return a * var_t{b, &a.t()}; }
auto operator/(var_t a, re_t b) { return a / var_t{b, &a.t()}; }
auto operator^(var_t a, re_t b) { return pow(a, var_t{b, &a.t()}); }

auto pow(re_t a, var_t b) { return pow(var_t{a, &b.t()}, b); }
auto operator+(re_t a, var_t b) { return var_t{a, &b.t()} + b; }
auto operator-(re_t a, var_t b) { return var_t{a, &b.t()} - b; }
auto operator*(re_t a, var_t b) { return var_t{a, &b.t()} * b; }
auto operator/(re_t a, var_t b) { return var_t{a, &b.t()} / b; }
auto operator^(re_t a, var_t b) { return pow(var_t{a, &b.t()}, b); }

var_t::var_t(re_t r, tape_t *t) {
  t_ = t;
  n_ = t_->new_variable(r).np();
}

re_t var_t::value() const { return n().v_; }

var_t &var_t::operator+=(var_t const &r) {
  (*this) = (*this) + r;
  return *this;
}

var_t &var_t::operator-=(var_t const &r) {
  (*this) = (*this) - r;
  return *this;
}

var_t &var_t::operator*=(var_t const &r) {
  (*this) = (*this) * r;
  return *this;
}

var_t &var_t::operator/=(var_t const &r) {
  (*this) = (*this) / r;
  return *this;
}

var_t var_t::operator-() const { return *this * var_t(-1.0, t_); }

var_t dot(vec_t<var_t> const &a, vec_t<var_t> const &b) {
  return v_b_to_1_op_mix<dot_mix>::fwd(a, b);
}

var_t asum(vec_t<var_t> const &a) { return v_u_to_1_op_mix<asum_mix>::fwd(a); }
var_t gsum(vec_t<var_t> const &a) { return v_u_to_1_op_mix<gsum_mix>::fwd(a); }

var_t mean(vec_t<var_t> const &a) { return asum(a) / re_t(a.size()); }
var_t gmean(vec_t<var_t> const &a) { return gsum(a) / re_t(a.size()); }

var_t max(vec_t<var_t> const &a) {
  assert(a.size());
  var_t m = a[0];
  for (size_t i = 1; i < a.size(); ++i) {
    if (a[i].value() > m.value()) {
      m = a[i];
    }
  }
  return m;
}

var_t min(vec_t<var_t> const &a) {
  assert(a.size());
  var_t m = a[0];
  for (size_t i = 1; i < a.size(); ++i) {
    if (a[i].value() < m.value()) {
      m = a[i];
    }
  }
  return m;
}

#pragma endregion

#pragma region backward

template <typename F> void backward_grad_accumulate(var_t &x, F df) {
  if (grad(x).is_alive()) {
    grad(x) += df();
  } else {
    grad(x) = df();
  }
}

void backward_set_grad_if_not_alive(var_t &x) {
  if (!grad(x).is_alive()) {
    grad(x) = x.t().new_variable(0.0);
  }
}

void sqrt_mix::backward(tape_t *t, node *n) {
  var_t f{t, n};
  var_t &fg = grad(f);
  var_t l{t, n->ls_.front()};

  auto df = [&]() { return 0.5 * fg / sqrt(l); };

  backward_grad_accumulate(l, df);
}

void exp_mix::backward(tape_t *t, node *n) {
  var_t f{t, n};
  var_t &fg = grad(f);
  var_t l{t, n->ls_.front()};

  auto df = [&]() { return f * fg; };

  backward_grad_accumulate(l, df);
}

void log_mix::backward(tape_t *t, node *n) {
  var_t f{t, n};
  var_t &fg = grad(f);
  var_t l{t, n->ls_.front()};

  auto df = [&]() { return fg / l; };

  backward_grad_accumulate(l, df);
}

void neg_mix::backward(tape_t *t, node *n) {
  var_t f{t, n};
  var_t &fg = grad(f);
  var_t l{t, n->ls_.front()};

  auto df = [&]() { return -fg; };

  backward_grad_accumulate(l, df);
}

void sin_mix::backward(tape_t *t, node *n) {

  var_t f{t, n};
  var_t &fg = grad(f);
  var_t l{t, n->ls_.front()};

  auto df = [&]() { return cos(l) * fg; };

  backward_grad_accumulate(l, df);
}

void cos_mix::backward(tape_t *t, node *n) {
  var_t f{t, n};
  var_t &fg = grad(f);
  var_t l{t, n->ls_.front()};

  auto df = [&]() { return -sin(l) * fg; };

  backward_grad_accumulate(l, df);
}

void tanh_mix::backward(tape_t *t, node *n) {

  var_t f{t, n};
  var_t &fg = grad(f);
  var_t l{t, n->ls_.front()};

  auto df = [&]() { return (1.0 - (f ^ 2.0)) * fg; };

  backward_grad_accumulate(l, df);
}

void relu_mix::backward(tape_t *t, node *n) {

  var_t f{t, n};
  var_t &fg = grad(f);
  var_t l{t, n->ls_.front()};

  auto df = [&]() { return (l.value() > 0.0) * fg; };

  backward_grad_accumulate(l, df);
}

// void id_mix::backward(tape_t *t, node *n) {
//   var_t f{t, n};
//   var_t &fg = grad(f);
//   var_t l{t, n->ls_.front()};
//
//   var_t one = t->new_variable(1.0);
//
//   if (!grad(l).is_alive()) {
//     grad(l) = var_t(0.0, t);
//   }
//
//   grad(l) += grad(l) + fg;
// }

void dot_mix::backward(tape_t *t, node *n) {
  var_t f{t, n};
  var_t &fg = grad(f);

  for (size_t i = 0; i < n->rs_.size(); ++i) {
    var_t l{t, n->ls_[i]};
    var_t r{t, n->rs_[i]};

    auto dfdl = [&]() { return r * fg; };
    auto dfdr = [&]() { return l * fg; };

    backward_grad_accumulate(l, dfdl);
    backward_grad_accumulate(r, dfdr);
  }
}

void asum_mix::backward(tape_t *t, node *n) {
  var_t f{t, n};
  var_t &fg = grad(f);

  for (auto &ls : n->ls_) {
    var_t l{t, ls};
    backward_set_grad_if_not_alive(l);
    grad(l) += fg;
  }
}

void add_mix::backward(tape_t *t, node *n) {
  var_t f{t, n};
  var_t &fg = grad(f);
  var_t l{t, n->ls_.front()};
  var_t r{t, n->rs_.front()};

  backward_set_grad_if_not_alive(l);
  backward_set_grad_if_not_alive(r);

  grad(l) += fg;
  grad(r) += fg;
}

void sub_mix::backward(tape_t *t, node *n) {
  var_t f{t, n};
  var_t &fg = grad(f);
  var_t l{t, n->ls_.front()};
  var_t r{t, n->rs_.front()};

  backward_set_grad_if_not_alive(l);
  backward_set_grad_if_not_alive(r);

  grad(l) += fg;
  grad(r) -= fg;
}

void mul_mix::backward(tape_t *t, node *n) {
  var_t f{t, n};
  var_t &fg = grad(f);
  var_t l{t, n->ls_.front()};
  var_t r{t, n->rs_.front()};

  auto dfdl = [&]() { return r * fg; };
  auto dfdr = [&]() { return l * fg; };

  backward_grad_accumulate(l, dfdl);
  backward_grad_accumulate(r, dfdr);
}

void div_mix::backward(tape_t *t, node *n) {
  var_t f{t, n};
  var_t &fg = grad(f);
  var_t l{t, n->ls_.front()};
  var_t r{t, n->rs_.front()};

  auto dfdl = [&]() { return fg / r; };
  auto dfdr = [&]() { return fg * l / (r * r); };

  backward_grad_accumulate(l, dfdl);

  if (grad(r).is_alive())
    grad(r) -= dfdr(); // negative sign
  else
    grad(r) = -dfdr();
}

void pow_mix::backward(tape_t *t, node *n) {
  var_t f{t, n};
  var_t &fg = grad(f);
  var_t l{t, n->ls_.front()};
  var_t r{t, n->rs_.front()};

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

void backward(var_t v) {

  v.t().fill_grads();

  if (!grad(v).is_alive()) {
    grad(v) = v.t().new_variable(1.0);
  }

  for (size_t idx = v.n().idx_ + 1; idx != 0; --idx) {
    node *n = &(v.t().nodes_[idx - 1]);
    var_t n_ = {&v.t(), n};
    if (!grad(n_).is_alive()) {
      grad(n_) = v.t().new_variable(0.0);
    }
    backward_impl(&n_.t(), &n_.n());
  }
}

#pragma endregion

} // namespace aks

namespace aks {

std::ostream &operator<<(std::ostream &o, var_t const &vs) {
  o << std::setprecision(15) << "var_t(" << vs.value() << ";"
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

inline std::string as_dot(var_t iv, size_t i, size_t cnt, size_t start_cnt) {
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
    std::string str = as_dot(var_t{&t, &data[idx]}, idx, cnt, start_count);
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
