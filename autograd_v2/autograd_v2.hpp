#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <deque>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
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

template <> struct is_vcl_vec<vcl::Vec4f> : std::true_type {};

template <> struct is_vcl_vec<vcl::Vec8f> : std::true_type {};

template <> struct is_vcl_vec<vcl::Vec16f> : std::true_type {};

template <> struct is_vcl_vec<vcl::Vec4fb> : std::true_type {};
template <> struct is_vcl_vec<vcl::Vec8fb> : std::true_type {};
template <> struct is_vcl_vec<vcl::Vec16fb> : std::true_type {};

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

template <typename F> vcl::Vec4f vec_for_each(F f, vcl::Vec4f const &x) {
  return vcl::Vec4f(f(x[0]), f(x[1]), f(x[2]), f(x[3]));
}

template <typename F> vcl::Vec8f vec_for_each(F f, vcl::Vec8f const &x) {
  return vcl::Vec8f(f(x[0]), f(x[1]), f(x[2]), f(x[3]), f(x[4]), f(x[5]),
                    f(x[6]), f(x[7]));
}

template <typename F> vcl::Vec16f vec_for_each(F f, vcl::Vec16f const &x) {
  return vcl::Vec16f(f(x[0]), f(x[1]), f(x[2]), f(x[3]), f(x[4]), f(x[5]),
                     f(x[6]), f(x[7]), f(x[8]), f(x[9]), f(x[10]), f(x[11]),
                     f(x[12]), f(x[13]), f(x[14]), f(x[15]));
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

template <typename F>
vcl::Vec4f vec_for_each(F f, vcl::Vec4f const &x, vcl::Vec4f const &y) {
  return vcl::Vec4f(f(x[0], y[0]), f(x[1], y[1]), f(x[2], y[2]), f(x[3], y[3]));
}

template <typename F>
vcl::Vec8f vec_for_each(F f, vcl::Vec8f const &x, vcl::Vec8f const &y) {
  return vcl::Vec8f(f(x[0], y[0]), f(x[1], y[1]), f(x[2], y[2]), f(x[3], y[3]),
                    f(x[4], y[4]), f(x[5], y[5]), f(x[6], y[6]), f(x[7], y[7]));
}

template <typename F>
vcl::Vec16f vec_for_each(F f, vcl::Vec16f const &x, vcl::Vec16f const &y) {
  return vcl::Vec16f(f(x[0], y[0]), f(x[1], y[1]), f(x[2], y[2]), f(x[3], y[3]),
                     f(x[4], y[4]), f(x[5], y[5]), f(x[6], y[6]), f(x[7], y[7]),
                     f(x[8], y[8]), f(x[9], y[9]), f(x[10], y[10]),
                     f(x[11], y[11]), f(x[12], y[12]), f(x[13], y[13]),
                     f(x[14], y[14]), f(x[15], y[15]));
}

AKS_ENABLE_FOR_VCL_VEC_1(T) auto vec_sin(const T &v) { return vcl::sin(v); }

AKS_ENABLE_FOR_VCL_VEC_1(T) auto vec_cos(const T &v) { return vcl::cos(v); };

AKS_ENABLE_FOR_VCL_VEC_1(T)
auto vec_tanh(const T &v) {
  return vec_for_each(
      [](const auto x) {
        using namespace std;
        return tanh(x);
      },
      v);
}

AKS_ENABLE_FOR_VCL_VEC_1(T) auto vec_exp(const T &v) { return vcl::exp(v); }

AKS_ENABLE_FOR_VCL_VEC_1(T) auto vec_log(const T &v) { return vcl::log(v); }

AKS_ENABLE_FOR_VCL_VEC_1(T)
auto vec_relu(const T &v) {
  return vec_for_each([](const auto a) { return (a > 0.0) ? a : 0.0; }, v);
}

AKS_ENABLE_FOR_VCL_VEC_1(T) auto vec_sqrt(const T &v) { return vcl::sqrt(v); }

AKS_ENABLE_FOR_VCL_VEC_1(T)
auto vec_pow(const T &v, const T &t) {
  return vcl::pow(v, t);
  // return vec_for_each([](const auto x, const auto y) { return std::pow(x, y);
  // },
  //                   v, t);
}
AKS_ENABLE_FOR_VCL_VEC_1(T) auto vec_max(const T &v, const T &t) {
  return vcl::max(v, t);
}
AKS_ENABLE_FOR_VCL_VEC_1(T) auto vec_min(const T &v, const T &t) {
  return vcl::min(v, t);
}
//
// std::string to_string(const vcl::Vec2d &v) {
//  std::stringstream os;
//  os << std::setprecision(15) << "(" << v[0] << ", " << v[1] << ")";
//  return os.str();
//}
//
// std::string to_string(const vcl::Vec4d &v) {
//  std::stringstream os;
//  os << std::setprecision(15) << "(" << v[0] << ", " << v[1] << ", " << v[2]
//     << ", " << v[3] << ")";
//  return os.str();
//}
//
// std::string to_string(const vcl::Vec8d &v) {
//  std::stringstream os;
//  os << std::setprecision(15) << "(" << v[0] << ", " << v[1] << ", " << v[2]
//     << ", " << v[3] << ", " << v[4] << ", " << v[5] << ", " << v[6] << ", "
//     << v[7] << ")";
//  return os.str();
//}

std::ostream &operator<<(std::ostream &os, const vcl::Vec4fb &v) {
  os << std::setprecision(15) << "(" << v[0] << ", " << v[1] << ", " << v[2]
     << ", " << v[3] << ")";
  return os;
}

std::ostream &operator<<(std::ostream &os, const vcl::Vec8fb &v) {
  os << std::setprecision(15) << "(" << v[0] << ", " << v[1] << ", " << v[2]
     << ", " << v[3] << ", " << v[4] << ", " << v[5] << ", " << v[6] << ", "
     << v[7] << ")";
  return os;
}

std::ostream &operator<<(std::ostream &os, const vcl::Vec16fb &v) {
  os << std::setprecision(15) << "(" << v[0] << ", " << v[1] << ", " << v[2]
     << ", " << v[3] << ", " << v[4] << ", " << v[5] << ", " << v[6] << ", "
     << v[7] << ", " << v[8] << ", " << v[9] << ", " << v[10] << ", " << v[11]
     << ", " << v[12] << ", " << v[13] << ", " << v[14] << ", " << v[15] << ")";
  return os;
}

std::ostream &operator<<(std::ostream &os, const vcl::Vec4f &v) {
  os << std::setprecision(15) << "(" << v[0] << ", " << v[1] << ", " << v[2]
     << ", " << v[3] << ")";
  return os;
}

std::ostream &operator<<(std::ostream &os, const vcl::Vec8f &v) {
  os << std::setprecision(15) << "(" << v[0] << ", " << v[1] << ", " << v[2]
     << ", " << v[3] << ", " << v[4] << ", " << v[5] << ", " << v[6] << ", "
     << v[7] << ")";
  return os;
}

std::ostream &operator<<(std::ostream &os, const vcl::Vec16f &v) {
  os << std::setprecision(15) << "(" << v[0] << ", " << v[1] << ", " << v[2]
     << ", " << v[3] << ", " << v[4] << ", " << v[5] << ", " << v[6] << ", "
     << v[7] << ", " << v[8] << ", " << v[9] << ", " << v[10] << ", " << v[11]
     << ", " << v[12] << ", " << v[13] << ", " << v[14] << ", " << v[15] << ")";
  return os;
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
template <typename K, typename V> using map_t = std::unordered_map<K, V>;
template <typename T> using set_t = std::unordered_set<T>;
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

template <typename F, typename T, typename... Ts>
auto zipped_op_in_place(F &&f, T &&t, Ts &&...ts) {
  for (size_t i = 0; i < t.size(); ++i) {
    f(t[i], ts[i]...);
  }
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

  var_t(value_type r, tape_type *t, bool requires_grad = false);

  var_t clone() const { return var_t{value(), t_, requires_grad()}; }

  value_type value() const;

  bool requires_grad() const { return n_ != nullptr && n_->requires_grad_; }

  tape_type &t() {
    assert(t_ != nullptr);
    return *t_;
  }
  tape_type &t() const { return *t_; }
  node_type &n() { return *n_; }
  node_type const &n() const {
    assert(is_alive());
    assert(n_ != nullptr);
    return *n_;
  }
  node_type *np() const { return n_; }

  var_t &operator=(var_t const &r) = default;
  var_t &operator=(var_t &&r) = default;

  var_t &operator+=(var_t const &r);
  var_t &operator-=(var_t const &r);
  var_t &operator*=(var_t const &r);
  var_t &operator/=(var_t const &r);
  var_t operator-() const;

  void update_in_place(value_type v, bool safe = true);

  // var_t& operator/=(var_t const& r);

  bool is_alive() const { return t_ != nullptr && n_ != nullptr; }

private:
  mutable tape_type *t_ = nullptr;
  node_type *n_ = nullptr;
}; // namespace aks

template <typename real_t> struct fwd_f {
  using value_type = real_t;
  typedef void (*fwd_f_t)(node<value_type> *);
  const char *n_ = nullptr;
  fwd_f_t f_ = nullptr;
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
  fwd_f<value_type> forwards_;
  vec_t<node<value_type> *> ls_, rs_;
  idx_t idx_ = sntnl_idx;
  bool requires_grad_ = false;
  bool is_leaf() const { return ls_.empty() && rs_.empty(); }
};

template <typename real_t> struct tape_t {
  using value_type = real_t;
  using node_type = node<value_type>;
  using grads_t = map_t<idx_t, var_t<value_type>>;

  node_type *new_node() {
    auto ret = &(nodes_.emplace_back());
    ret->idx_ = nodes_.size() - 1;
    return ret;
  }

  var_t<value_type> new_variable(value_type const &r,
                                 bool requires_grad = false) {
    node_type *n = new_node();
    n->v_ = r;
    n->requires_grad_ = requires_grad;
    return {this, n};
  }

  void fill_grads() {
    // grads_.resize(nodes_.size());
  }

  void zero_grad() { grads_.clear(); }

  void keep_only(std::tuple<idx_t, grads_t> const &ds) {
    nodes_.resize(std::get<0>(ds));
    grads_ = std::get<1>(ds);
  }

  void reset() {
    keep_only({0, grads_t{}});
    stack_t<std::tuple<idx_t, grads_t>> clear;
    saved_state_.swap(clear);
  }
  void push_state() { saved_state_.push({nodes_.size(), grads_}); }

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
  grads_t grads_;
  stack_t<std::tuple<idx_t, grads_t>> saved_state_;
};

template <typename real_t> struct tape_context {
  tape_context(tape_t<real_t> &tape) : tape_(tape) { tape_.push_state(); }
  ~tape_context() { tape_.pop_state(); }
  tape_t<real_t> &tape_;
};

template <typename real_t> auto &grad_unchecked(var_t<real_t> &n) {
  return n.t().grads_[n.n().idx_];
}

template <typename real_t> auto const &grad_unchecked(var_t<real_t> const &n) {
  return n.t().grads_[n.n().idx_];
}

template <typename real_t> auto &grad(var_t<real_t> &n, bool safe = true) {
  assert(n.requires_grad());
  assert(n.is_alive());
  if (!n.requires_grad()) {
    real_t nan_ = std::numeric_limits<real_t>::quiet_NaN();
    n.t().grads_[n.n().idx_] = n.t().new_variable(nan_);
  }
  if (safe) {
    // in safe mode - you can only grad at leaf node
    assert(n.is_alive() && n.n().is_leaf());
  }
  if (!n.t().grads_.contains(n.n().idx_)) {
    n.t().grads_[n.n().idx_] = n.t().new_variable(real_t(0.0));
  }
  return grad_unchecked(n);
}

template <typename real_t>
auto const &grad(var_t<real_t> const &n, bool safe = true) {
  assert(n.requires_grad());
  assert(n.is_alive());
  if (!n.requires_grad()) {
    real_t nan_ = std::numeric_limits<real_t>::quiet_NaN();
    n.t().grads_[n.n().idx_] = n.t().new_variable(nan_);
  }
  if (safe) {
    // in safe mode - you can only grad at leaf node
    assert(n.is_alive() && n.n().is_leaf());
  }
  if (!n.t().grads_.contains(n.n().idx_)) {
    n.t().grads_[n.n().idx_] = n.t().new_variable(0.0);
  }
  return grad_unchecked(n);
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
      CHECK_NAN(sqrt(a));
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
      // return vec_relu(a);
      return vcl_select(a > value_type(0.0), a, value_type(0.0));
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
  // static void backward(tape_t<value_type> *t, node<value_type> *n);
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

template <typename mix> struct u_op_mix : mix {
  using mix::apply;
  using mix::bf;
  using value_type = mix::value_type;
  static value_type apply_op(value_type const &a) { return apply(a); }
  static void forward(node<value_type> *n) {
    n->v_ = apply_op(n->ls_.front()->v_);
  }
  static var_t<value_type> fwd_onto_tape(var_t<value_type> const &a) {
    node<value_type> *new_node = a.t().new_node();
    new_node->backwards_ = bf();
    new_node->forwards_ = {bf().n_, forward};
    new_node->ls_.push_back(a.np());
    new_node->v_ = apply_op(a.n().v_);
    new_node->requires_grad_ = a.n().requires_grad_;
    return var_t{&a.t(), new_node};
  }
};

template <typename real_t>
struct u_op_mix<relu_mix<real_t>> : relu_mix<real_t> {
  using mix = relu_mix<real_t>;
  using mix::apply;
  using mix::bf;
  using value_type = mix::value_type;
  static value_type apply_op(value_type const &a) { return apply(a); }
  static void forward(node<value_type> *n) {
    if (!n->rs_.empty()) {
      // then we have had a backward run we want to update
      if constexpr (aks::vcl_detail::is_vcl_vec<value_type>::value) {
        using namespace ::aks::vcl_detail;
        n->rs_.front()->v_ = vcl_select(n->ls_.front()->v_ < value_type(0.0),
                                        value_type(0.0), value_type(1.0));
      } else {
        n->rs_.front()->v_ = n->ls_.front()->v_ < value_type(0.0)
                                 ? value_type(0.0)
                                 : value_type(1.0);
      }
    }
    n->v_ = apply_op(n->ls_.front()->v_);
  }
  static var_t<value_type> fwd_onto_tape(var_t<value_type> const &a) {
    node<value_type> *new_node = a.t().new_node();
    new_node->backwards_ = bf();
    new_node->forwards_ = {bf().n_, forward};
    new_node->ls_.push_back(a.np());
    new_node->v_ = apply_op(a.n().v_);
    new_node->requires_grad_ = a.n().requires_grad_;
    return var_t{&a.t(), new_node};
  }
};

template <typename mix> struct op_mix : mix {
  using mix::apply;
  using mix::bf;
  using value_type = mix::value_type;
  static value_type apply_op(value_type const &a, value_type const &b) {
    return apply(a, b);
  }
  static void forward(node<value_type> *n) {
    n->v_ = apply_op(n->ls_.front()->v_, n->rs_.front()->v_);
  }
  static var_t<value_type> fwd_onto_tape(var_t<value_type> const &a,
                                         var_t<value_type> const &b) {
    node<value_type> *new_node = a.t().new_node();
    new_node->backwards_ = bf();
    new_node->forwards_ = {bf().n_, forward};
    new_node->ls_.push_back(a.np());
    new_node->rs_.push_back(b.np());
    new_node->v_ = apply_op(a.n().v_, b.n().v_);
    new_node->requires_grad_ = a.n().requires_grad_ || b.n().requires_grad_;
    return var_t{&a.t(), new_node};
  }
};

template <typename mix> struct v_u_to_1_op_mix : mix {
  using mix::apply;
  using mix::bf;
  using mix::start;
  using value_type = mix::value_type;
  static void forward(node<value_type> *n) {
    n->v_ = start();
    for (int i = 0; i < n->ls_.size(); i++) {
      apply(n->v_, n->ls_[i]->v_);
    }
  }
  static std::tuple<value_type, bool>
  fwd_impl(vec_t<var_t<value_type>> const &a, vec_t<node<value_type> *> &ls) {
    value_type ret = start();
    bool requires_grad = false;
    for (int i = 0; i < a.size(); i++) {
      apply(ret, a[i].n().v_);
      ls.push_back(a[i].np());
      requires_grad = requires_grad || a[i].n().requires_grad_;
    }
    return {ret, requires_grad};
  }
  static var_t<value_type> fwd_onto_tape(vec_t<var_t<value_type>> const &a) {
    assert(a.size());
    node<value_type> *new_node = a.front().t().new_node();
    std::tie(new_node->v_, new_node->requires_grad_) =
        fwd_impl(a, new_node->ls_);
    new_node->backwards_ = bf();
    new_node->forwards_ = {bf().n_, forward};
    return var_t{&a.front().t(), new_node};
  }
};

template <typename mix> struct v_b_to_1_op_mix : mix {
  using mix::apply;
  using mix::bf;
  using mix::start;
  using value_type = mix::value_type;
  static void forward(node<value_type> *n) {
    n->v_ = start();
    for (int i = 0; i < n->ls_.size(); i++) {
      apply(n->v_, n->ls_[i]->v_, n->rs_[i]->v_);
    }
  }

  static std::tuple<value_type, bool>
  fwd_impl(vec_t<var_t<value_type>> const &a, vec_t<var_t<value_type>> const &b,
           vec_t<node<value_type> *> &ls, vec_t<node<value_type> *> &rs) {
    value_type ret = start();
    assert(a.size() == b.size());
    bool requires_grad = false;
    for (int i = 0; i < a.size(); i++) {
      apply(ret, a[i].n().v_, b[i].n().v_);
      ls.push_back(a[i].np());
      rs.push_back(b[i].np());
      requires_grad =
          requires_grad || a[i].n().requires_grad_ || b[i].n().requires_grad_;
    }
    return {ret, requires_grad};
  }

  static var_t<value_type> fwd_onto_tape(vec_t<var_t<value_type>> const &a,
                                         vec_t<var_t<value_type>> const &b) {
    assert(a.size());
    assert(a.size() == b.size());
    node<value_type> *new_node = a.front().t().new_node();
    std::tie(new_node->v_, new_node->requires_grad_) =
        fwd_impl(a, b, new_node->ls_, new_node->rs_);
    new_node->backwards_ = bf();
    new_node->forwards_ = {bf().n_, forward};
    return var_t{&a.front().t(), new_node};
  }
};

#pragma endregion

#pragma region operator_overloads

// auto identity(var_t<real_t> a) { return u_op_mix<id_mix>::fwd(a); }

template <typename real_t> auto relu(var_t<real_t> a) {
  return u_op_mix<relu_mix<real_t>>::fwd_onto_tape(a);
}
template <typename real_t> auto sqrt(var_t<real_t> a) {
  return u_op_mix<sqrt_mix<real_t>>::fwd_onto_tape(a);
}

template <typename real_t> auto sin(var_t<real_t> a) {
  return u_op_mix<sin_mix<real_t>>::fwd_onto_tape(a);
}
template <typename real_t> auto cos(var_t<real_t> a) {
  return u_op_mix<cos_mix<real_t>>::fwd_onto_tape(a);
}
template <typename real_t> auto tanh(var_t<real_t> a) {
  return u_op_mix<tanh_mix<real_t>>::fwd_onto_tape(a);
}
template <typename real_t> auto neg(var_t<real_t> a) {
  return u_op_mix<neg_mix<real_t>>::fwd_onto_tape(a);
}

template <typename real_t> auto exp(var_t<real_t> a) {
  return u_op_mix<exp_mix<real_t>>::fwd_onto_tape(a);
}
template <typename real_t> auto log(var_t<real_t> a) {
  return u_op_mix<log_mix<real_t>>::fwd_onto_tape(a);
}

template <typename real_t> auto pow(var_t<real_t> a, var_t<real_t> b) {
  return op_mix<pow_mix<real_t>>::fwd_onto_tape(a, b);
}
template <typename real_t> auto pow(var_t<real_t> a, real_t b) {
  return pow(a, var_t<real_t>{b, &a.t()});
}
template <typename real_t> auto pow(real_t a, var_t<real_t> b) {
  return pow(var_t<real_t>{a, &b.t()}, b);
}
template <typename real_t> auto operator+(var_t<real_t> a, var_t<real_t> b) {
  return op_mix<add_mix<real_t>>::fwd_onto_tape(a, b);
}
template <typename real_t> auto operator-(var_t<real_t> a, var_t<real_t> b) {
  return op_mix<sub_mix<real_t>>::fwd_onto_tape(a, b);
}
template <typename real_t> auto operator*(var_t<real_t> a, var_t<real_t> b) {
  return op_mix<mul_mix<real_t>>::fwd_onto_tape(a, b);
}
template <typename real_t> auto operator/(var_t<real_t> a, var_t<real_t> b) {
  return op_mix<div_mix<real_t>>::fwd_onto_tape(a, b);
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

template <typename real_t>
var_t<real_t>::var_t(real_t r, tape_t<real_t> *t, bool requires_grad) {
  t_ = t;
  n_ = t_->new_variable(r, requires_grad).np();
}

template <typename real_t>
void var_t<real_t>::update_in_place(real_t v, bool safe) {
  if (safe) {
    // can only update a leaf node
    // not sure why a non-safe would ever be handy... but oh well
    assert(n().is_leaf());
  }
  n().v_ = v;
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
  return *this * var_t(real_t(-1.0), t_);
}

template <typename real_t>
var_t<real_t> dot(vec_t<var_t<real_t>> const &a,
                  vec_t<var_t<real_t>> const &b) {
  return v_b_to_1_op_mix<dot_mix<real_t>>::fwd_onto_tape(a, b);
}

template <typename real_t> var_t<real_t> asum(vec_t<var_t<real_t>> const &a) {
  return v_u_to_1_op_mix<asum_mix<real_t>>::fwd_onto_tape(a);
}

template <typename real_t> var_t<real_t> gsum(vec_t<var_t<real_t>> const &a) {
  return v_u_to_1_op_mix<gsum_mix<real_t>>::fwd_onto_tape(a);
}

template <typename real_t> var_t<real_t> mean(vec_t<var_t<real_t>> const &a) {
  return asum(a) / static_cast<real_t>(static_cast<double>(a.size()));
}

template <typename real_t> var_t<real_t> gmean(vec_t<var_t<real_t>> const &a) {
  return gsum(a) / static_cast<real_t>(static_cast<double>(a.size()));
}

template <typename real_t>
std::enable_if<vcl_detail::is_vcl_vec<real_t>::value, var_t<real_t>>::type
max(vec_t<var_t<real_t>> const &a) {
  assert(false);
  return {};
  /*
  var_t<vcl::Vec2d> m = a[0];
  for (size_t i = 1; i < a.size(); ++i) {
    m = vcl_detail::vec_max(m, a[i]);
  }
  return m;*/
}

template <typename real_t>
std::enable_if<!vcl_detail::is_vcl_vec<real_t>::value, var_t<real_t>>::type
max(vec_t<var_t<real_t>> const &a) {
  assert(a.size());
  var_t<real_t> m = a[0];
  for (size_t i = 1; i < a.size(); ++i) {
    if (a[i].value() > m.value()) {
      m = a[i];
    }
  }
  return m;
}

template <typename real_t>
std::enable_if<vcl_detail::is_vcl_vec<real_t>::value, var_t<real_t>>::type
min(vec_t<var_t<real_t>> const &a) {
  assert(false);
  return {}; /*
   var_t<vcl::Vec2d> m = a[0];
   for (size_t i = 1; i < a.size(); ++i) {
     m = vcl_detail::vec_min(m, a[i]);
   }
   return m;*/
}

template <typename real_t>
std::enable_if<!vcl_detail::is_vcl_vec<real_t>::value, var_t<real_t>>::type
min(vec_t<var_t<real_t>> const &a) {
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

template <typename real_t>
void forward_impl(tape_t<real_t> *t, var_t<real_t> const *from,
                  var_t<real_t> const *upto, bool safe) {
  if (safe && from) {
    // in safe mode - you can only start from a leaf node
    // else if you are doing partial runs, you can start anywhere
    assert(from->n().is_leaf());
  }

  // we do not want to recalc from, as upto will do that
  size_t i = from ? from->np()->idx_ + 1 : 0;
  for (; i < t->nodes_.size(); ++i) {
    if (upto && upto->np()->idx_ < i) {
      return;
    }
    node<real_t> &n = t->nodes_[i];
    if (n.forwards_.f_) {
      n.forwards_.f_(&n);
    }
  }
}

template <typename real_t> void forward(tape_t<real_t> *t) {
  forward_impl<real_t>(t, nullptr, nullptr, true);
}

template <typename real_t>
void forward_from(tape_t<real_t> *t, var_t<real_t> const *from,
                  bool safe = true) {
  forward_impl<real_t>(t, from, nullptr, safe);
}

template <typename real_t>
void forward_to(tape_t<real_t> *t, var_t<real_t> const *upto) {
  forward_impl<real_t>(t, nullptr, upto, true);
}

template <typename real_t>
void forward_from_to(tape_t<real_t> *t, var_t<real_t> const *from,
                     var_t<real_t> const *upto, bool safe = true) {
  forward_impl<real_t>(t, from, upto, safe);
}

#pragma region backward

template <typename real_t, typename F>
void backward_grad_accumulate(var_t<real_t> &x, F df) {
  if (x.n().requires_grad_) {
    if (grad_unchecked(x).is_alive()) {
      grad_unchecked(x) += df();
    } else {
      grad_unchecked(x) = df();
    }
  }
}

template <typename real_t>
void backward_set_grad_if_not_alive(var_t<real_t> &x) {
  if (x.n().requires_grad_) {
    if (!grad_unchecked(x).is_alive()) {
      grad_unchecked(x) = x.t().new_variable(real_t(0.0));
    }
  }
}

template <typename real_t>
void sqrt_mix<real_t>::backward(tape_t<real_t> *t, node<real_t> *n) {
  var_t<real_t> f{t, n};
  var_t<real_t> &fg = grad_unchecked(f);
  var_t<real_t> l{t, n->ls_.front()};

  auto df = [&]() { return real_t{0.5} * fg / sqrt(l); };

  backward_grad_accumulate(l, df);
}

template <typename real_t>
void exp_mix<real_t>::backward(tape_t<real_t> *t, node<real_t> *n) {
  var_t<real_t> f{t, n};
  var_t<real_t> &fg = grad_unchecked(f);
  var_t<real_t> l{t, n->ls_.front()};

  auto df = [&]() { return f * fg; };

  backward_grad_accumulate(l, df);
}

template <typename real_t>
void log_mix<real_t>::backward(tape_t<real_t> *t, node<real_t> *n) {
  var_t<real_t> f{t, n};
  var_t<real_t> &fg = grad_unchecked(f);
  var_t<real_t> l{t, n->ls_.front()};

  auto df = [&]() { return fg / l; };

  backward_grad_accumulate(l, df);
}

template <typename real_t>
void neg_mix<real_t>::backward(tape_t<real_t> *t, node<real_t> *n) {
  var_t<real_t> f{t, n};
  var_t<real_t> &fg = grad_unchecked(f);
  var_t<real_t> l{t, n->ls_.front()};

  auto df = [&]() { return -fg; };

  backward_grad_accumulate(l, df);
}

template <typename real_t>
void sin_mix<real_t>::backward(tape_t<real_t> *t, node<real_t> *n) {

  var_t<real_t> f{t, n};
  var_t<real_t> &fg = grad_unchecked(f);
  var_t<real_t> l{t, n->ls_.front()};

  auto df = [&]() { return cos(l) * fg; };

  backward_grad_accumulate(l, df);
}

template <typename real_t>
void cos_mix<real_t>::backward(tape_t<real_t> *t, node<real_t> *n) {
  var_t<real_t> f{t, n};
  var_t<real_t> &fg = grad_unchecked(f);
  var_t<real_t> l{t, n->ls_.front()};

  auto df = [&]() { return -sin(l) * fg; };

  backward_grad_accumulate(l, df);
}

template <typename real_t>
void tanh_mix<real_t>::backward(tape_t<real_t> *t, node<real_t> *n) {

  var_t<real_t> f{t, n};
  var_t<real_t> &fg = grad_unchecked(f);
  var_t<real_t> l{t, n->ls_.front()};

  auto df = [&]() { return (real_t{1.0} - (f ^ real_t{2.0})) * fg; };

  backward_grad_accumulate(l, df);
}

template <typename real_t>
void relu_mix<real_t>::backward(tape_t<real_t> *t, node<real_t> *n) {

  var_t<real_t> f{t, n};
  var_t<real_t> &fg = grad_unchecked(f);
  var_t<real_t> l{t, n->ls_.front()};

  auto df = [&]() {
    if constexpr (aks::vcl_detail::is_vcl_vec<real_t>::value) {
      using namespace ::aks::vcl_detail;
      real_t selection =
          vcl_select(l.value() > real_t{0.0}, real_t{1.0}, real_t{0.0});
      var_t<real_t> sel = t->new_variable(selection);
      n->rs_.push_back(sel.np());
      return sel * fg;
    } else {
      real_t selection = real_t(l.value() > real_t{0.0});
      var_t<real_t> sel = t->new_variable(selection);
      n->rs_.push_back(sel.np());
      return sel * fg;
    }
  };

  backward_grad_accumulate(l, df);
}

// template<typename real_t> void id_mix<real_t>::backward(tape_t<real_t>*t,
// node<real_t>*n) {
//   var_t<real_t> f{t, n};
//   var_t<real_t> &fg = grad_unchecked(f);
//   var_t<real_t> l{t, n->ls_.front()};
//
//   var_t<real_t> one = t->new_variable(1.0);
//
//   if (!grad_unchecked(l).is_alive()) {
//     grad_unchecked(l) = var_t(0.0, t);
//   }
//
//   grad_unchecked(l) += grad_unchecked(l) + fg;
// }

template <typename real_t>
void dot_mix<real_t>::backward(tape_t<real_t> *t, node<real_t> *n) {
  var_t<real_t> f{t, n};
  var_t<real_t> &fg = grad_unchecked(f);

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
  var_t<real_t> &fg = grad_unchecked(f);

  for (auto &ls : n->ls_) {
    var_t<real_t> l{t, ls};
    if (l.n().requires_grad_) {
      backward_set_grad_if_not_alive(l);
      grad_unchecked(l) += fg;
    }
  }
}

template <typename real_t>
void add_mix<real_t>::backward(tape_t<real_t> *t, node<real_t> *n) {
  var_t<real_t> f{t, n};
  var_t<real_t> &fg = grad_unchecked(f);
  var_t<real_t> l{t, n->ls_.front()};
  var_t<real_t> r{t, n->rs_.front()};

  if (l.requires_grad()) {
    backward_set_grad_if_not_alive(l);
    grad_unchecked(l) += fg;
  }

  if (r.requires_grad()) {
    backward_set_grad_if_not_alive(r);
    grad_unchecked(r) += fg;
  }
}

template <typename real_t>
void sub_mix<real_t>::backward(tape_t<real_t> *t, node<real_t> *n) {
  var_t<real_t> f{t, n};
  var_t<real_t> &fg = grad_unchecked(f);
  var_t<real_t> l{t, n->ls_.front()};
  var_t<real_t> r{t, n->rs_.front()};

  if (l.requires_grad()) {
    backward_set_grad_if_not_alive(l);
    grad_unchecked(l) += fg;
  }

  if (r.requires_grad()) {
    backward_set_grad_if_not_alive(r);
    grad_unchecked(r) -= fg;
  }
}

template <typename real_t>
void mul_mix<real_t>::backward(tape_t<real_t> *t, node<real_t> *n) {
  var_t<real_t> f{t, n};
  var_t<real_t> &fg = grad_unchecked(f);
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
  var_t<real_t> &fg = grad_unchecked(f);
  var_t<real_t> l{t, n->ls_.front()};
  var_t<real_t> r{t, n->rs_.front()};

  auto dfdl = [&]() { return fg / r; };
  auto dfdr = [&]() { return fg * l / (r * r); };

  backward_grad_accumulate(l, dfdl);

  if (grad_unchecked(r).is_alive())
    grad_unchecked(r) -= dfdr(); // negative sign
  else
    grad_unchecked(r) = -dfdr();
}

template <typename real_t>
void pow_mix<real_t>::backward(tape_t<real_t> *t, node<real_t> *n) {
  var_t<real_t> f{t, n};
  var_t<real_t> &fg = grad_unchecked(f);
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

  if (!grad_unchecked(v).is_alive()) {
    grad_unchecked(v) = v.t().new_variable(real_t(1.0));
  }

  stack_t<node<real_t> *> stk;
  set_t<idx_t> seen;

  stk.push(&(v.n()));

  while (!stk.empty()) {
    node<real_t> *n = stk.top();
    stk.pop();
    if (!seen.contains(n->idx_)) {
      seen.insert(n->idx_);
      for (auto &l : n->ls_) {
        if (l->requires_grad_) {
          stk.push(l);
        }
      }
      for (auto &r : n->rs_) {
        if (r->requires_grad_) {
          stk.push(r);
        }
      }
    }
  }

  vec_t<idx_t> idxs(seen.begin(), seen.end());

  std::sort(idxs.begin(), idxs.end(), std::greater<idx_t>());

  for (idx_t idx : idxs) {
    node<real_t> *n = &(v.t().nodes_[idx]);
    var_t<real_t> n_ = {&v.t(), n};
    if (!grad_unchecked(n_).is_alive()) {
      grad_unchecked(n_) = v.t().new_variable(real_t(0.0));
    }
    backward_impl(&n_.t(), &n_.n());
  }
}

#pragma endregion

} // namespace aks

namespace aks {

template <typename real_t>
std::ostream &operator<<(std::ostream &o, var_t<real_t> const &vs) {
  using namespace vcl_detail;
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
  using namespace aks::vcl_detail;
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
    return x == "add" || x == "sub" || x == "mul" || x == "div" || x == "pow" ||
           x == "relu";
  };
  auto is_unary_op = [](std::string const &x) {
    return x == "neg" || x == "tanh" || x == "sin" || x == "cos" ||
           x == "identity" || x == "sqrt" || x == "log" || x == "exp";
  };
  auto is_binary_vec_op = [](std::string const &x) { return x == "dot"; };

  if (!nd.is_leaf()) {
    ss << "\n"
       << idx << " [label = \"[" << idx << "] " << std::setprecision(15)
       << to_string(nd) << "\", fontcolor=blue, color=blue];\n";
    if (is_binary_op(nd.backwards_.n_)) {
      if (check(nd.ls_.front())) {
        ss << nd.ls_.front()->idx_ + start_cnt << "->" << nd.backwards_.n_
           << cnt << "; ";
      }
      if ((!nd.rs_.empty()) && check(nd.rs_.front())) {
        ss << nd.rs_.front()->idx_ + start_cnt << "->" << nd.backwards_.n_
           << cnt << "; ";
      }
      if (check(nd.ls_.front()) ||
          ((!nd.rs_.empty()) && check(nd.rs_.front()))) {
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
    } else if (is_binary_vec_op(nd.backwards_.n_)) {
      if (check(nd.ls_.front())) {
        auto const &children0 = nd.ls_;
        for (auto const &c : children0) {
          // if (check(c)) {
          ss << c->idx_ << "->" << nd.backwards_.n_ << cnt << ";\n";
          //}
        }
      }
      if (check(nd.rs_.front())) {
        auto const &children1 = nd.rs_;
        for (auto const &c : children1) {
          // if (check(c)) {
          ss << c->idx_ << "->" << nd.backwards_.n_ << cnt << ";\n";
          //}
        }
      }
      if (check(nd.ls_.front()) || check(nd.rs_.front())) {
        ss << nd.backwards_.n_ << cnt << "->" << idx << ";\n";
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

namespace std {
template <typename real_t> struct hash<aks::var_t<real_t>> {
  size_t operator()(const aks::var_t<real_t> &v) const {
    return std::hash<aks::idx_t>()(v.n().idx_);
  }
};

template <typename real_t> struct equal_to<aks::var_t<real_t>> {
  size_t operator()(const aks::var_t<real_t> &v,
                    const aks::var_t<real_t> &w) const {
    return v.n().idx_ == w.n().idx_;
  }
};
} // namespace std

namespace aks {
template <typename real_t>
vec_t<var_t<real_t>> put_on_tape(tape_t<real_t> *tape, vec_t<real_t> const &xs,
                                 bool requires_grad = false) {
  vec_t<var_t<real_t>> ret;
  ret.reserve(xs.size());
  for (auto const &x : xs) {
    ret.push_back(tape->new_variable(x, requires_grad));
  }
  return ret;
}

template <typename real_t>
vec_t<real_t> get_values(vec_t<var_t<real_t>> const &xs) {
  vec_t<real_t> ret;
  ret.reserve(xs.size());
  for (auto const &x : xs) {
    assert(x.is_alive());
    ret.push_back(x.value());
  }
  return ret;
}

template <typename real_t>
vec_t<var_t<real_t>> get_grads(vec_t<var_t<real_t>> const &xs) {
  vec_t<var_t<real_t>> ret;
  ret.reserve(xs.size());
  for (auto const &x : xs) {
    assert(x.is_alive());
    assert(x.requires_grad());
    ret.push_back(grad(x));
  }
  return ret;
}

template <typename real_t>
map_t<var_t<real_t>, var_t<real_t>>
get_grads_map(vec_t<var_t<real_t>> const &xs) {
  map_t<var_t<real_t>, var_t<real_t>> ret;
  ret.reserve(xs.size());
  for (auto const &x : xs) {
    assert(x.is_alive());
    assert(x.requires_grad());
    ret[x] = grad(x);
  }
  return ret;
}

template <typename real_t, typename... var_ts>
auto get_grads(var_t<real_t> x, var_ts... xs) {
  return std::make_tuple(grad(x), grad(xs)...);
}

template <typename real_t> void build_gradients(var_t<real_t> f) {
  assert(f.is_alive());
  assert(f.requires_grad());
  f.t().zero_grad();
  backward(f);
}

template <typename real_t>
var_t<real_t> gradient(var_t<real_t> f, var_t<real_t> x) {
  build_gradients(f);
  return grad(x);
}

template <typename real_t, typename... vars>
auto gradient(var_t<real_t> f, var_t<real_t> x, vars... xs) {
  build_gradients(f);
  return get_grads(x, xs...);
}

template <typename real_t>
vec_t<var_t<real_t>> gradient(var_t<real_t> f, vec_t<var_t<real_t>> const &xs) {
  build_gradients(f);
  return get_grads(xs);
}

template <typename real_t>
vec_t<vec_t<var_t<real_t>>> gradient(var_t<real_t> f,
                                     vec_t<vec_t<var_t<real_t>>> const &xss) {
  build_gradients(f);
  vec_t<vec_t<var_t<real_t>>> ret;
  ret.reserve(xss.size());
  for (auto const &xs : xss) {
    ret.emplace_back(get_grads(xs));
  }
  return ret;
}

template <typename real_t>
vec_t<var_t<real_t>> gradient(vec_t<var_t<real_t>> const &fs,
                              vec_t<var_t<real_t>> const &xs) {
  vec_t<var_t<real_t>> ret;
  ret.reserve(fs.size());
  assert(fs.size() == xs.size());
  for (size_t i = 0; i < fs.size(); ++i) {
    ret.push_back(gradient(fs[i], xs[i]));
  }
  return ret;
}

template <typename real_t>
map_t<var_t<real_t>, var_t<real_t>>
gradient_map(var_t<real_t> f, vec_t<var_t<real_t>> const &xs) {
  build_gradients(f);
  return get_grads_map(xs);
}

template <typename real_t>
map_t<var_t<real_t>, var_t<real_t>>
gradient_map(vec_t<var_t<real_t>> const &fs, vec_t<var_t<real_t>> const &xs) {
  map_t<var_t<real_t>, var_t<real_t>> ret;

  assert(fs.size() == xs.size());
  for (size_t i = 0; i < fs.size(); ++i) {
    ret[xs[i]] = gradient(fs[i], xs[i]);
  }
  return ret;
}

template <typename real_t>
var_t<real_t> higher_order_gradient(var_t<real_t> f, var_t<real_t> x,
                                    size_t order) {
  for (size_t i = 0; i < order; ++i) {
    f = gradient(f, x);
  }
  return f;
}

template <typename real_t>
var_t<real_t> higher_order_gradient(
    var_t<real_t> f, std::vector<std::tuple<var_t<real_t>, size_t>> const &xs) {
  for (auto const &[x, order] : xs) {
    f = higher_order_gradient(f, x, order);
  }
  return f;
}

} // namespace aks
