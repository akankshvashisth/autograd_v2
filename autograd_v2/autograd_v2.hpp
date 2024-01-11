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
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

#ifndef AKS_NO_VCL

#define VCL_NAMESPACE vcl

#include <fog/vectorclass.h>
#include <fog/vectormath_common.h>
#include <fog/vectormath_exp.h>
// #include <fog/vectormath_hyp.h>
// #include <fog/vectormath_lib.h>
#include <fog/vectormath_trig.h>

#endif

namespace aks {
namespace vcl_detail {
template <typename T> struct is_vcl_vec : std::false_type {};

#ifndef AKS_NO_VCL

#define AKS_ENABLE_FOR_VCL_VEC_1(T)                                            \
  template <typename T, std::enable_if_t<is_vcl_vec<T>::value, int> = true>

template <> struct is_vcl_vec<vcl::Vec2d> : std::true_type {};

template <> struct is_vcl_vec<vcl::Vec4d> : std::true_type {};

template <> struct is_vcl_vec<vcl::Vec8d> : std::true_type {};

template <typename F> vcl::Vec2d vec_for_each(F f, vcl::Vec2d const &x) {
  return vcl::Vec2d(f(x[0]), f(x[1]));
}

template <typename F> vcl::Vec4d vec_for_each(F f, vcl::Vec4d const &x) {
  return vcl::Vec4d(f(x[0]), f(x[1]), f(x[2]), f(x[3]));
}

template <typename F> vcl::Vec8d vec_for_each(F f, vcl::Vec8d const &x) {
  return vcl::Vec8d(f(x[0]), f(x[1]), f(x[2]), f(x[3]), f(x[4]), f(x[5]),
                    f(x[6]), f(x[7]));
}

template <typename F>
vcl::Vec2d vec_for_each(F f, vcl::Vec2d const &x, vcl::Vec2d const &y) {
  return vcl::Vec2d(f(x[0], y[0]), f(x[1], y[1]));
}

template <typename F>
vcl::Vec4d vec_for_each(F f, vcl::Vec4d const &x, vcl::Vec4d const &y) {
  return vcl::Vec4d(f(x[0], y[0]), f(x[1], y[1]), f(x[2], y[2]), f(x[3], y[3]));
}

template <typename F>
vcl::Vec8d vec_for_each(F f, vcl::Vec8d const &x, vcl::Vec8d const &y) {
  return vcl::Vec8d(f(x[0], y[0]), f(x[1], y[1]), f(x[2], y[2]), f(x[3], y[3]),
                    f(x[4], y[4]), f(x[5], y[5]), f(x[6], y[6]), f(x[7], y[7]));
}

AKS_ENABLE_FOR_VCL_VEC_1(T) auto vec_sin(const T &v) { return vcl::sin(v); }

AKS_ENABLE_FOR_VCL_VEC_1(T) auto vec_cos(const T &v) { return vcl::cos(v); };

AKS_ENABLE_FOR_VCL_VEC_1(T)
auto vec_tanh(const T &v) {
  return vec_for_each([](const double x) { return std::tanh(x); }, v);
}

AKS_ENABLE_FOR_VCL_VEC_1(T) auto vec_exp(const T &v) { return vcl::exp(v); }

AKS_ENABLE_FOR_VCL_VEC_1(T) auto vec_log(const T &v) { return vcl::log(v); }

AKS_ENABLE_FOR_VCL_VEC_1(T)
auto vec_relu(const T &v) {
  return vec_for_each([](const double a) { return (a > 0.0) ? a : 0.0; }, v);
}

AKS_ENABLE_FOR_VCL_VEC_1(T) auto vec_sqrt(const T &v) { return vcl::sqrt(v); }

AKS_ENABLE_FOR_VCL_VEC_1(T)
auto vec_pow(const T &v, const T &t) {
  return vec_for_each(
      [](const double x, const double y) { return std::pow(x, y); }, v, t);
}
AKS_ENABLE_FOR_VCL_VEC_1(T) auto vec_max(const T &v, const T &t) {
  return vcl::max(v, t);
}
AKS_ENABLE_FOR_VCL_VEC_1(T) auto vec_min(const T &v, const T &t) {
  return vcl::min(v, t);
}

std::string to_string(const vcl::Vec2d &v) {
  std::stringstream os;
  os << std::setprecision(15) << "(" << v[0] << ", " << v[1] << ")";
  return os.str();
}

std::ostream &operator<<(std::ostream &os, const vcl::Vec2d &v) {
  os << std::setprecision(15) << "(" << v[0] << ", " << v[1] << ")";
  return os;
}

std::ostream &operator<<(std::ostream &os, const vcl::Vec4d &v) {
  os << std::setprecision(15) << "(" << v[0] << ", " << v[1] << ", " << v[2]
     << ", " << v[3] << ")";
  return os;
}

std::ostream &operator<<(std::ostream &os, const vcl::Vec8d &v) {
  os << std::setprecision(15) << "(" << v[0] << ", " << v[1] << ", " << v[2]
     << ", " << v[3] << ", " << v[4] << ", " << v[5] << ", " << v[6] << ", "
     << v[7] << ")";
  return os;
}

template <typename R, typename T, typename U>
auto vcl_select(typename R &&s, T &&l, U &&r) {
  return vcl::select(std::forward<R>(s), std::forward<T>(l),
                     std::forward<U>(r));
}

template <typename T> auto vec_horizontal_or(T &&v) {
  return vcl::horizontal_or(std::forward<T>(v));
}

#endif

template <typename R, typename T> R get_first_as(T const &v) {
  if constexpr (is_vcl_vec<T>::value) {
    return R(v[0]);
  } else {
    return R(v);
  }
}

} // namespace vcl_detail

} // namespace aks

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

template <typename real_t> struct tape_t;

template <typename real_t> struct node;

template <typename real_t> struct var_t {
  using value_type = real_t;
  using tape_type = tape_t<real_t>;
  using node_type = node<real_t>;

  var_t(tape_type *t, node_type *n) : t_(t), n_(n) {}
  var_t() : t_(nullptr), n_(nullptr) {}

  var_t(var_t const &r) = default;
  var_t(var_t &&r) = default;

  var_t(value_type r, tape_type *t);

  var_t clone() const { return var_t{value(), t_}; }

  value_type value() const;

  tape_type &t() { return *t_; }
  tape_type &t() const { return *t_; }
  node_type &n() { return *n_; }
  node_type const &n() const { return *n_; }
  node_type *np() const { return n_; }

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
  mutable tape_type *t_ = nullptr;
  node_type *n_ = nullptr;
};

template <typename real_t> struct back_f {
  using value_type = real_t;
  typedef void (*back_f_t)(tape_t<value_type> *, node<value_type> *);
  const char *n_ = nullptr;
  back_f_t f_ = nullptr;
};

template <typename real_t> struct node {
  using value_type = real_t;
  value_type v_ = value_type{};
  back_f<value_type> backwards_;
  vec_t<node<value_type> *> ls_, rs_;
  idx_t idx_ = sntnl_idx;
  bool is_leaf() const { return ls_.empty() && rs_.empty(); }
};

template <typename real_t> struct tape_t {
  using value_type = real_t;
  using node_type = node<value_type>;

  node_type *new_node() {
    auto ret = &(nodes_.emplace_back());
    ret->idx_ = nodes_.size() - 1;
    return ret;
  }

  var_t<value_type> new_variable(value_type const &r) {
    node_type *n = new_node();
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

  deq_t<node_type> nodes_;
  vec_t<var_t<value_type>> grads_;
  stack_t<std::tuple<idx_t, idx_t>> saved_state_;
};

template <typename real_t> auto &grad(var_t<real_t> &n) {
  return n.t().grads_[n.n().idx_];
}

template <typename real_t> auto const &grad(var_t<real_t> const &n) {
  return n.t().grads_[n.n().idx_];
}

#pragma endregion

#pragma region mixes

template <typename real_t> struct exp_mix {
  using value_type = real_t;
  static value_type apply(value_type a) {
    if constexpr (::aks::vcl_detail::is_vcl_vec<value_type>::value) {
      using namespace ::aks::vcl_detail;
      return vec_exp(a);
    } else {
      using namespace std;
      CHECK_NAN(exp(a));
    }
  }
  static back_f<value_type> bf() { return {"exp", backward}; }
  static void backward(tape_t<value_type> *t, node<value_type> *n);
};

template <typename real_t> struct log_mix {
  using value_type = real_t;
  static value_type apply(value_type a) {

    if constexpr (::aks::vcl_detail::is_vcl_vec<value_type>::value) {
      using namespace ::aks::vcl_detail;
      return vec_log(a);
    } else {
      using namespace std;
      CHECK_NAN(log(a));
    }
  }
  static back_f<value_type> bf() { return {"log", backward}; }
  static void backward(tape_t<value_type> *t, node<value_type> *n);
};

template <typename real_t> struct tanh_mix {
  using value_type = real_t;
  static value_type apply(value_type a) {

    if constexpr (::aks::vcl_detail::is_vcl_vec<value_type>::value) {
      using namespace ::aks::vcl_detail;
      return vec_tanh(a);
    } else {
      using namespace std;
      CHECK_NAN(tanh(a));
    }
  }
  static back_f<value_type> bf() { return {"tanh", backward}; }
  static void backward(tape_t<value_type> *t, node<value_type> *n);
};

template <typename real_t> struct neg_mix {
  using value_type = real_t;
  static value_type apply(value_type a) { return -a; }
  static back_f<value_type> bf() { return {"neg", backward}; }
  static void backward(tape_t<value_type> *t, node<value_type> *n);
};

template <typename real_t> struct sin_mix {
  using value_type = real_t;
  static value_type apply(value_type a) {
    if constexpr (::aks::vcl_detail::is_vcl_vec<value_type>::value) {
      using namespace ::aks::vcl_detail;
      return vec_sin(a);
    } else {
      using namespace std;
      CHECK_NAN(sin(a));
    }
  }
  static back_f<value_type> bf() { return {"sin", backward}; }
  static void backward(tape_t<value_type> *t, node<value_type> *n);
};

template <typename real_t> struct cos_mix {
  using value_type = real_t;
  static value_type apply(value_type a) {

    if constexpr (::aks::vcl_detail::is_vcl_vec<value_type>::value) {
      using namespace ::aks::vcl_detail;
      return vec_cos(a);
    } else {
      using namespace std;
      CHECK_NAN(cos(a));
    }
  }
  static back_f<value_type> bf() { return {"cos", backward}; }
  static void backward(tape_t<value_type> *t, node<value_type> *n);
};

template <typename real_t> struct sqrt_mix {
  using value_type = real_t;
  static value_type apply(value_type a) {

    if constexpr (::aks::vcl_detail::is_vcl_vec<value_type>::value) {
      using namespace ::aks::vcl_detail;
      return vec_sqrt(a);
    } else {
      using namespace std;
      CHECK_NAN(std::sqrt(a));
    }
  }

  static back_f<value_type> bf() { return {"sqrt", backward}; }
  static void backward(tape_t<value_type> *t, node<value_type> *n);
};

template <typename real_t> struct relu_mix {
  using value_type = real_t;
  static value_type apply(value_type a) {
    if constexpr (::aks::vcl_detail::is_vcl_vec<value_type>::value) {
      using namespace ::aks::vcl_detail;
      return vec_relu(a);
    } else {
      using namespace std;
      return (a > value_type(0.0)) ? a : value_type(0.0);
    }
  }

  static back_f<value_type> bf() { return {"relu", backward}; }
  static void backward(tape_t<value_type> *t, node<value_type> *n);
};

// struct id_mix {
//   static value_type apply(value_type a) { return a; }
//   static back_f<value_type> bf() { return {"id", backward}; }
//   static void backward(tape_t<value_type>*t, node<value_type>*n);
// };

template <typename real_t> struct add_mix {
  using value_type = real_t;
  static value_type apply(value_type a, value_type b) { return a + b; }
  static back_f<value_type> bf() { return {"add", backward}; }
  static void backward(tape_t<value_type> *t, node<value_type> *n);
};

template <typename real_t> struct sub_mix {
  using value_type = real_t;
  static value_type apply(value_type a, value_type b) { return a - b; }
  static back_f<value_type> bf() { return {"sub", backward}; }
  static void backward(tape_t<value_type> *t, node<value_type> *n);
};

template <typename real_t> struct mul_mix {
  using value_type = real_t;
  static value_type apply(value_type a, value_type b) { return a * b; }
  static back_f<value_type> bf() { return {"mul", backward}; }
  static void backward(tape_t<value_type> *t, node<value_type> *n);
};

template <typename real_t> struct div_mix {
  using value_type = real_t;
  static value_type apply(value_type a, value_type b) { return a / b; }
  static back_f<value_type> bf() { return {"div", backward}; }
  static void backward(tape_t<value_type> *t, node<value_type> *n);
};

template <typename real_t> struct pow_mix {
  using value_type = real_t;
  static value_type apply(value_type a, value_type b) {
    if constexpr (aks::vcl_detail::is_vcl_vec<value_type>::value) {
      using namespace ::aks::vcl_detail;
      return vec_pow(a, b);
    } else {
      using namespace std;
      CHECK_NAN(pow(a, b));
    }
  }
  static back_f<value_type> bf() { return {"pow", backward}; }
  static void backward(tape_t<value_type> *t, node<value_type> *n);
};

template <typename mix> struct u_op_mix : mix {
  using mix::apply;
  using mix::bf;
  using value_type = mix::value_type;
  static value_type fwd(value_type const &a) { return apply(a); }
  static var_t<value_type> fwd(var_t<value_type> const &a) {
    node<value_type> *new_node = a.t().new_node();
    new_node->v_ = fwd(a.n().v_);
    new_node->backwards_ = bf();
    new_node->ls_.push_back(a.np());
    return var_t{&a.t(), new_node};
  }
};

template <typename mix> struct op_mix : mix {
  using mix::apply;
  using mix::bf;
  using value_type = mix::value_type;
  static value_type fwd(value_type const &a, value_type const &b) {
    return apply(a, b);
  }
  static var_t<value_type> fwd(var_t<value_type> const &a,
                               var_t<value_type> const &b) {
    node<value_type> *new_node = a.t().new_node();
    new_node->v_ = fwd(a.n().v_, b.n().v_);
    new_node->backwards_ = bf();
    new_node->ls_.push_back(a.np());
    new_node->rs_.push_back(b.np());
    return var_t{&a.t(), new_node};
  }
};

template <typename real_t> struct dot_mix {
  using value_type = real_t;
  static value_type start() { return 0.0; }
  static void apply(value_type &r, value_type a, value_type b) { r += (a * b); }
  static back_f<value_type> bf() { return {"dot", backward}; }
  static void backward(tape_t<value_type> *t, node<value_type> *n);
};

template <typename real_t> struct asum_mix {
  using value_type = real_t;
  static value_type start() { return 0.0; }
  static void apply(value_type &r, value_type a) { r += a; }
  static back_f<value_type> bf() { return {"asum", backward}; }
  static void backward(tape_t<value_type> *t, node<value_type> *n);
};

template <typename real_t> struct gsum_mix {
  using value_type = real_t;
  static value_type start() { return 1.0; }
  static void apply(value_type &r, value_type a) { r *= a; }
  static back_f<value_type> bf() { return {"gsum", nullptr}; }
  static void backward(tape_t<value_type> *t, node<value_type> *n);
};

// template <typename real_t>
//  struct max_mix {
//    static value_type start() { return -INFINITY; }
//    static void apply(value_type &r, value_type a) { r = std::max(r, a); }
//    static back_f<value_type> bf() { return {"max", nullptr}; }
//    // static void backward(tape_t* t, node* n);
//  };
//
//  template <typename real_t>
//  struct min_mix {
//    static value_type start() { return INFINITY; }
//    static void apply(value_type &r, value_type a) { r = std::min(r, a); }
//    static back_f<value_type> bf() { return {"min", nullptr}; }
//    // static void backward(tape_t* t, node* n);
//  };

template <typename mix> struct v_u_to_1_op_mix : mix {
  using mix::apply;
  using mix::bf;
  using mix::start;
  using value_type = mix::value_type;
  static value_type fwd_impl(vec_t<var_t<value_type>> const &a,
                             vec_t<node<value_type> *> &ls) {
    value_type ret = start();
    for (int i = 0; i < a.size(); i++) {
      apply(ret, a[i].n().v_);
      ls.push_back(a[i].np());
    }
    return ret;
  }
  static var_t<value_type> fwd(vec_t<var_t<value_type>> const &a) {
    assert(a.size());
    node<value_type> *new_node = a.front().t().new_node();
    new_node->v_ = fwd_impl(a, new_node->ls_);
    new_node->backwards_ = bf();
    return var_t{&a.front().t(), new_node};
  }
};

template <typename mix> struct v_b_to_1_op_mix : mix {
  using mix::apply;
  using mix::bf;
  using mix::start;
  using value_type = mix::value_type;
  static value_type fwd_impl(vec_t<var_t<value_type>> const &a,
                             vec_t<var_t<value_type>> const &b,
                             vec_t<node<value_type> *> &ls,
                             vec_t<node<value_type> *> &rs) {
    value_type ret = start();
    for (int i = 0; i < a.size(); i++) {
      apply(ret, a[i].n().v_, b[i].n().v_);
      ls.push_back(a[i].np());
      rs.push_back(b[i].np());
    }
    return ret;
  }
  static var_t<value_type> fwd(vec_t<var_t<value_type>> const &a,
                               vec_t<var_t<value_type>> const &b) {
    assert(a.size());
    assert(a.size() == b.size());
    node<value_type> *new_node = a.front().t().new_node();
    new_node->v_ = fwd_impl(a, b, new_node->ls_, new_node->rs_);
    new_node->backwards_ = bf();
    return var_t{&a.front().t(), new_node};
  }
};

#pragma endregion

#pragma region operator_overloads

// auto identity(var_t<real_t> a) { return u_op_mix<id_mix>::fwd(a); }

template <typename real_t> auto relu(var_t<real_t> a) {
  return u_op_mix<relu_mix<real_t>>::fwd(a);
}
template <typename real_t> auto sqrt(var_t<real_t> a) {
  return u_op_mix<sqrt_mix<real_t>>::fwd(a);
}

template <typename real_t> auto sin(var_t<real_t> a) {
  return u_op_mix<sin_mix<real_t>>::fwd(a);
}
template <typename real_t> auto cos(var_t<real_t> a) {
  return u_op_mix<cos_mix<real_t>>::fwd(a);
}
template <typename real_t> auto tanh(var_t<real_t> a) {
  return u_op_mix<tanh_mix<real_t>>::fwd(a);
}
template <typename real_t> auto neg(var_t<real_t> a) {
  return u_op_mix<neg_mix<real_t>>::fwd(a);
}

template <typename real_t> auto exp(var_t<real_t> a) {
  return u_op_mix<exp_mix<real_t>>::fwd(a);
}
template <typename real_t> auto log(var_t<real_t> a) {
  return u_op_mix<log_mix<real_t>>::fwd(a);
}

template <typename real_t> auto pow(var_t<real_t> a, var_t<real_t> b) {
  return op_mix<pow_mix<real_t>>::fwd(a, b);
}
template <typename real_t> auto pow(var_t<real_t> a, real_t b) {
  return pow(a, var_t<real_t>{b, &a.t()});
}
template <typename real_t> auto pow(real_t a, var_t<real_t> b) {
  return pow(var_t<real_t>{a, &b.t()}, b);
}
template <typename real_t> auto operator+(var_t<real_t> a, var_t<real_t> b) {
  return op_mix<add_mix<real_t>>::fwd(a, b);
}
template <typename real_t> auto operator-(var_t<real_t> a, var_t<real_t> b) {
  return op_mix<sub_mix<real_t>>::fwd(a, b);
}
template <typename real_t> auto operator*(var_t<real_t> a, var_t<real_t> b) {
  return op_mix<mul_mix<real_t>>::fwd(a, b);
}
template <typename real_t> auto operator/(var_t<real_t> a, var_t<real_t> b) {
  return op_mix<div_mix<real_t>>::fwd(a, b);
}
template <typename real_t> auto operator^(var_t<real_t> a, var_t<real_t> b) {
  return pow(a, b);
}
template <typename real_t> auto operator+(var_t<real_t> a, real_t b) {
  return a + var_t{b, &a.t()};
}
template <typename real_t> auto operator-(var_t<real_t> a, real_t b) {
  return a - var_t{b, &a.t()};
}
template <typename real_t> auto operator*(var_t<real_t> a, real_t b) {
  return a * var_t{b, &a.t()};
}
template <typename real_t> auto operator/(var_t<real_t> a, real_t b) {
  return a / var_t{b, &a.t()};
}
template <typename real_t> auto operator^(var_t<real_t> a, real_t b) {
  return pow(a, var_t{b, &a.t()});
}
template <typename real_t> auto operator+(real_t a, var_t<real_t> b) {
  return var_t{a, &b.t()} + b;
}
template <typename real_t> auto operator-(real_t a, var_t<real_t> b) {
  return var_t{a, &b.t()} - b;
}
template <typename real_t> auto operator*(real_t a, var_t<real_t> b) {
  return var_t{a, &b.t()} * b;
}
template <typename real_t> auto operator/(real_t a, var_t<real_t> b) {
  return var_t{a, &b.t()} / b;
}
template <typename real_t> auto operator^(real_t a, var_t<real_t> b) {
  return pow(var_t{a, &b.t()}, b);
}

template <typename real_t> var_t<real_t>::var_t(real_t r, tape_t<real_t> *t) {
  t_ = t;
  n_ = t_->new_variable(r).np();
}

template <typename real_t> real_t var_t<real_t>::value() const {
  return n().v_;
}

template <typename real_t>
var_t<real_t> &var_t<real_t>::operator+=(var_t<real_t> const &r) {
  (*this) = (*this) + r;
  return *this;
}

template <typename real_t>
var_t<real_t> &var_t<real_t>::operator-=(var_t<real_t> const &r) {
  (*this) = (*this) - r;
  return *this;
}

template <typename real_t>
var_t<real_t> &var_t<real_t>::operator*=(var_t<real_t> const &r) {
  (*this) = (*this) * r;
  return *this;
}

template <typename real_t>
var_t<real_t> &var_t<real_t>::operator/=(var_t<real_t> const &r) {
  (*this) = (*this) / r;
  return *this;
}

template <typename real_t> var_t<real_t> var_t<real_t>::operator-() const {
  return *this * var_t(-1.0, t_);
}

template <typename real_t>
var_t<real_t> dot(vec_t<var_t<real_t>> const &a,
                  vec_t<var_t<real_t>> const &b) {
  return v_b_to_1_op_mix<dot_mix<real_t>>::fwd(a, b);
}

template <typename real_t> var_t<real_t> asum(vec_t<var_t<real_t>> const &a) {
  return v_u_to_1_op_mix<asum_mix<real_t>>::fwd(a);
}

template <typename real_t> var_t<real_t> gsum(vec_t<var_t<real_t>> const &a) {
  return v_u_to_1_op_mix<gsum_mix<real_t>>::fwd(a);
}

template <typename real_t> var_t<real_t> mean(vec_t<var_t<real_t>> const &a) {
  return asum(a) / static_cast<real_t>(static_cast<double>(a.size()));
}

template <typename real_t> var_t<real_t> gmean(vec_t<var_t<real_t>> const &a) {
  return gsum(a) / static_cast<real_t>(static_cast<double>(a.size()));
}

#ifndef AKS_NO_VCL

var_t<vcl::Vec2d> max(vec_t<var_t<vcl::Vec2d>> const &a) {
  assert(false);
  return {};
  /*
  var_t<vcl::Vec2d> m = a[0];
  for (size_t i = 1; i < a.size(); ++i) {
    m = vcl_detail::vec_max(m, a[i]);
  }
  return m;*/
}

var_t<vcl::Vec2d> min(vec_t<var_t<vcl::Vec2d>> const &a) {
  assert(false);
  return {}; /*
   var_t<vcl::Vec2d> m = a[0];
   for (size_t i = 1; i < a.size(); ++i) {
     m = vcl_detail::vec_min(m, a[i]);
   }
   return m;*/
}

#endif

template <typename real_t> var_t<real_t> max(vec_t<var_t<real_t>> const &a) {
  assert(a.size());
  var_t<real_t> m = a[0];
  for (size_t i = 1; i < a.size(); ++i) {
    if (a[i].value() > m.value()) {
      m = a[i];
    }
  }
  return m;
}

template <typename real_t> var_t<real_t> min(vec_t<var_t<real_t>> const &a) {
  assert(a.size());
  var_t<real_t> m = a[0];
  for (size_t i = 1; i < a.size(); ++i) {
    if (a[i].value() < m.value()) {
      m = a[i];
    }
  }
  return m;
}

#pragma endregion

#pragma region backward

template <typename real_t, typename F>
void backward_grad_accumulate(var_t<real_t> &x, F df) {
  if (grad(x).is_alive()) {
    grad(x) += df();
  } else {
    grad(x) = df();
  }
}

template <typename real_t>
void backward_set_grad_if_not_alive(var_t<real_t> &x) {
  if (!grad(x).is_alive()) {
    grad(x) = x.t().new_variable(0.0);
  }
}

template <typename real_t>
void sqrt_mix<real_t>::backward(tape_t<real_t> *t, node<real_t> *n) {
  var_t<real_t> f{t, n};
  var_t<real_t> &fg = grad(f);
  var_t<real_t> l{t, n->ls_.front()};

  auto df = [&]() { return real_t{0.5} * fg / sqrt(l); };

  backward_grad_accumulate(l, df);
}

template <typename real_t>
void exp_mix<real_t>::backward(tape_t<real_t> *t, node<real_t> *n) {
  var_t<real_t> f{t, n};
  var_t<real_t> &fg = grad(f);
  var_t<real_t> l{t, n->ls_.front()};

  auto df = [&]() { return f * fg; };

  backward_grad_accumulate(l, df);
}

template <typename real_t>
void log_mix<real_t>::backward(tape_t<real_t> *t, node<real_t> *n) {
  var_t<real_t> f{t, n};
  var_t<real_t> &fg = grad(f);
  var_t<real_t> l{t, n->ls_.front()};

  auto df = [&]() { return fg / l; };

  backward_grad_accumulate(l, df);
}

template <typename real_t>
void neg_mix<real_t>::backward(tape_t<real_t> *t, node<real_t> *n) {
  var_t<real_t> f{t, n};
  var_t<real_t> &fg = grad(f);
  var_t<real_t> l{t, n->ls_.front()};

  auto df = [&]() { return -fg; };

  backward_grad_accumulate(l, df);
}

template <typename real_t>
void sin_mix<real_t>::backward(tape_t<real_t> *t, node<real_t> *n) {

  var_t<real_t> f{t, n};
  var_t<real_t> &fg = grad(f);
  var_t<real_t> l{t, n->ls_.front()};

  auto df = [&]() { return cos(l) * fg; };

  backward_grad_accumulate(l, df);
}

template <typename real_t>
void cos_mix<real_t>::backward(tape_t<real_t> *t, node<real_t> *n) {
  var_t<real_t> f{t, n};
  var_t<real_t> &fg = grad(f);
  var_t<real_t> l{t, n->ls_.front()};

  auto df = [&]() { return -sin(l) * fg; };

  backward_grad_accumulate(l, df);
}

template <typename real_t>
void tanh_mix<real_t>::backward(tape_t<real_t> *t, node<real_t> *n) {

  var_t<real_t> f{t, n};
  var_t<real_t> &fg = grad(f);
  var_t<real_t> l{t, n->ls_.front()};

  auto df = [&]() { return (real_t{1.0} - (f ^ real_t{2.0})) * fg; };

  backward_grad_accumulate(l, df);
}

template <typename real_t>
void relu_mix<real_t>::backward(tape_t<real_t> *t, node<real_t> *n) {

  var_t<real_t> f{t, n};
  var_t<real_t> &fg = grad(f);
  var_t<real_t> l{t, n->ls_.front()};

  auto df = [&]() {
    if constexpr (aks::vcl_detail::is_vcl_vec<real_t>::value) {
      using namespace ::aks::vcl_detail;
      return vcl_select(l.value() > 0.0, real_t{1.0}, real_t{0.0}) * fg;
    } else {
      return real_t(l.value() > 0.0) * fg;
    }
  };

  backward_grad_accumulate(l, df);
}

// template<typename real_t> void id_mix<real_t>::backward(tape_t<real_t>*t,
// node<real_t>*n) {
//   var_t<real_t> f{t, n};
//   var_t<real_t> &fg = grad(f);
//   var_t<real_t> l{t, n->ls_.front()};
//
//   var_t<real_t> one = t->new_variable(1.0);
//
//   if (!grad(l).is_alive()) {
//     grad(l) = var_t(0.0, t);
//   }
//
//   grad(l) += grad(l) + fg;
// }

template <typename real_t>
void dot_mix<real_t>::backward(tape_t<real_t> *t, node<real_t> *n) {
  var_t<real_t> f{t, n};
  var_t<real_t> &fg = grad(f);

  for (size_t i = 0; i < n->rs_.size(); ++i) {
    var_t<real_t> l{t, n->ls_[i]};
    var_t<real_t> r{t, n->rs_[i]};

    auto dfdl = [&]() { return r * fg; };
    auto dfdr = [&]() { return l * fg; };

    backward_grad_accumulate(l, dfdl);
    backward_grad_accumulate(r, dfdr);
  }
}

template <typename real_t>
void asum_mix<real_t>::backward(tape_t<real_t> *t, node<real_t> *n) {
  var_t<real_t> f{t, n};
  var_t<real_t> &fg = grad(f);

  for (auto &ls : n->ls_) {
    var_t<real_t> l{t, ls};
    backward_set_grad_if_not_alive(l);
    grad(l) += fg;
  }
}

template <typename real_t>
void add_mix<real_t>::backward(tape_t<real_t> *t, node<real_t> *n) {
  var_t<real_t> f{t, n};
  var_t<real_t> &fg = grad(f);
  var_t<real_t> l{t, n->ls_.front()};
  var_t<real_t> r{t, n->rs_.front()};

  backward_set_grad_if_not_alive(l);
  backward_set_grad_if_not_alive(r);

  grad(l) += fg;
  grad(r) += fg;
}

template <typename real_t>
void sub_mix<real_t>::backward(tape_t<real_t> *t, node<real_t> *n) {
  var_t<real_t> f{t, n};
  var_t<real_t> &fg = grad(f);
  var_t<real_t> l{t, n->ls_.front()};
  var_t<real_t> r{t, n->rs_.front()};

  backward_set_grad_if_not_alive(l);
  backward_set_grad_if_not_alive(r);

  grad(l) += fg;
  grad(r) -= fg;
}

template <typename real_t>
void mul_mix<real_t>::backward(tape_t<real_t> *t, node<real_t> *n) {
  var_t<real_t> f{t, n};
  var_t<real_t> &fg = grad(f);
  var_t<real_t> l{t, n->ls_.front()};
  var_t<real_t> r{t, n->rs_.front()};

  auto dfdl = [&]() { return r * fg; };
  auto dfdr = [&]() { return l * fg; };

  backward_grad_accumulate(l, dfdl);
  backward_grad_accumulate(r, dfdr);
}

template <typename real_t>
void div_mix<real_t>::backward(tape_t<real_t> *t, node<real_t> *n) {
  var_t<real_t> f{t, n};
  var_t<real_t> &fg = grad(f);
  var_t<real_t> l{t, n->ls_.front()};
  var_t<real_t> r{t, n->rs_.front()};

  auto dfdl = [&]() { return fg / r; };
  auto dfdr = [&]() { return fg * l / (r * r); };

  backward_grad_accumulate(l, dfdl);

  if (grad(r).is_alive())
    grad(r) -= dfdr(); // negative sign
  else
    grad(r) = -dfdr();
}

template <typename real_t>
void pow_mix<real_t>::backward(tape_t<real_t> *t, node<real_t> *n) {
  var_t<real_t> f{t, n};
  var_t<real_t> &fg = grad(f);
  var_t<real_t> l{t, n->ls_.front()};
  var_t<real_t> r{t, n->rs_.front()};

  auto dfdl = [&]() { return fg * (r * pow(l, r - real_t(1.0))); };
  auto dfdr = [&]() { return fg * f * log(l); };

  backward_grad_accumulate(l, dfdl);
  backward_grad_accumulate(r, dfdr);
}

template <typename real_t>
void backward_impl(tape_t<real_t> *t, node<real_t> *n) {
  if (n->backwards_.f_ != nullptr) {
    n->backwards_.f_(t, n);
  }
}

template <typename real_t> void backward(var_t<real_t> v) {

  v.t().fill_grads();

  if (!grad(v).is_alive()) {
    grad(v) = v.t().new_variable(1.0);
  }

  for (size_t idx = v.n().idx_ + 1; idx != 0; --idx) {
    node<real_t> *n = &(v.t().nodes_[idx - 1]);
    var_t<real_t> n_ = {&v.t(), n};
    if (!grad(n_).is_alive()) {
      grad(n_) = v.t().new_variable(0.0);
    }
    backward_impl(&n_.t(), &n_.n());
  }
}

#pragma endregion

} // namespace aks

namespace aks {
#ifndef AKS_NO_VCL

std::ostream &operator<<(std::ostream &o, var_t<vcl::Vec2d> const &vs) {
  using namespace vcl_detail;
  o << std::setprecision(15) << "var_t(" << vs.value() << ";"
    << (vs.n().backwards_.n_ ? vs.n().backwards_.n_ : "null") << ";"
    << vs.t().nodes_.size() << ")";
  return o;
}
#endif

template <typename real_t>
std::ostream &operator<<(std::ostream &o, var_t<real_t> const &vs) {
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

template <typename real_t> std::string to_string(node<real_t> const &v) {
  std::stringstream ss;
  ss << v.v_;
  return ss.str();
}

template <typename real_t>
inline std::string as_dot(var_t<real_t> iv, size_t i, size_t cnt,
                          size_t start_cnt) {
  auto check = [](auto x) { return x != nullptr; };

  std::stringstream ss;

  size_t idx = i + start_cnt;
  node<real_t> &nd = iv.n();

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

template <typename real_t>
inline std::string as_dot(tape_t<real_t> &t, size_t start_count = 0) {
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
    const node<real_t> &d = data.at(idx);
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
