
#if 0

#include <iostream>

#include <array>
#include <deque>
#include <limits>
#include <stack>
#include <string>
#include <tuple>
#include <vector>

namespace aks {

#define AKS_PRINT(EXPR) std::cout << #EXPR << " = " << EXPR << std::endl

#define AKS_PRINT_AS(NAME, EXPR)                                               \
  std::cout << std::setprecision(15) << NAME << " " << EXPR << "       \t("    \
            << #EXPR << ")" << std::endl

template <typename T> using vec = std::vector<T>;
template <typename T> using stk = std::stack<T, std::vector<T>>;
using r_t = double;
using reals_t = vec<r_t>;
using idx_t = size_t;
using idxs_t = vec<idx_t>;
using idx_p_t = std::array<idx_t, 2>;
constexpr idx_t const sntnl_idx = std::numeric_limits<idx_t>::max();
constexpr idx_p_t const sntnl_idx_p = {sntnl_idx, sntnl_idx};

struct tp_dt;
struct tape_t;

typedef void (*back_f_t)(tape_t *, idx_t);

struct back_f {
  const char *n_;
  back_f_t f_;
};

template <bool g> struct v_t {
  v_t(r_t r, tape_t *t);
  v_t() : t_(nullptr), i_(sntnl_idx) {}

  static v_t preexisting(tape_t *t, idx_t i) { return v_t(t, i); }

  v_t &operator=(r_t r);

  tape_t *t() const { return t_; }
  idx_t i() const { return i_; }
  auto cs() const;
  v_t g() const;
  bool has_g() const;
  // void set_g(v_t const &r);
  void detach() {
    t_ = nullptr;
    i_ = sntnl_idx;
  }

private:
  v_t(tape_t *t, idx_t i) : t_(t), i_(i) {}
  mutable tape_t *t_;
  idx_t i_;
};

struct tp_dt {
  tp_dt(r_t v, idx_t g, idx_p_t const &cs, back_f bf)
      : v_(v), g_(g), cs_(cs), bf_(bf) {}
  tp_dt() : tp_dt(0, sntnl_idx, sntnl_idx_p, {nullptr, nullptr}) {}
  tp_dt(r_t v) : tp_dt(v, sntnl_idx, sntnl_idx_p, {nullptr, nullptr}) {}
  tp_dt(r_t v, idx_p_t const &cs, back_f bf) : tp_dt(v, sntnl_idx, cs, bf) {}

  tp_dt(tp_dt &&other) = default;
  tp_dt(tp_dt const &other) = delete;

  tp_dt &operator=(tp_dt &&other) = default;
  tp_dt &operator=(tp_dt const &other) = delete;

  bool is_leaf() const { return cs_[0] == sntnl_idx; }
  bool has_grad() const { return g_ != sntnl_idx; }
  tp_dt &get_grad(tape_t *t);
  tp_dt const &get_grad(tape_t *t) const;

  r_t v_;
  idx_t g_;
  idx_p_t cs_;
  back_f bf_;
};

struct tape_t {
  tape_t() : d_(), vcs_(), sp_() {}
  auto &at(idx_t const &i) { return d_[i]; }
  auto &v_at(idx_t const &i) { return d_[i].v_; }
  idx_t emplace(tp_dt &&t) {
    d_.emplace_back(std::move(t));
    return d_.size() - 1;
  }
  idx_t emplace(idxs_t &&vcs) {
    vcs_.emplace_back(std::move(vcs));
    return vcs_.size() - 1;
  }
  void clear_grads(idx_t i) {
    for (auto it = d_.begin(); it != d_.begin() + i; ++it) {
      if (it->has_grad())
        it->g_ = sntnl_idx;
    }
  }

  template <bool g> void clear_grads(v_t<g> i) { clear_grads(i.i() + 1); }

  void clear_grads() { clear_grads(d_.size()); }

  auto get_data() const { return std::forward_as_tuple(d_, vcs_); }

  void keep_only(std::array<uint64_t, 2> size) {
    d_.resize(size[0]);
    vcs_.resize(size[1]);
  }

  size_t size() { return d_.size(); }

  void reserve(size_t s) { d_.reserve(s); }

  void reset() {
    keep_only({0, 0});
    stk<idx_p_t> clear;
    sp_.swap(clear);
  }

  void push_store_point() { sp_.push({d_.size(), vcs_.size()}); }
  void pop_store_point() {
    if (!sp_.empty()) {
      keep_only(sp_.top());
      sp_.pop();
    } else {
      reset();
    }
  }

private:
  vec<tp_dt> d_;
  vec<idxs_t> vcs_;
  stk<idx_p_t> sp_;
};

tp_dt &tp_dt::get_grad(tape_t *t) { return t->at(g_); }
tp_dt const &tp_dt::get_grad(tape_t *t) const { return t->at(g_); }

template <bool g> v_t<g>::v_t(r_t r, tape_t *t) : t_(t), i_(t->emplace(r)) {}

template <bool g> v_t<g> &v_t<g>::operator=(r_t r) {
  // if (t_->at(i_).is_leaf()) {
  t_->at(i_).v_ = r;
  if (t_->at(i_).has_grad())
    t_->at(i_).get_grad(t_).v_ = 0.0;
  // keep the child and backward function as is.
  return *this;
}
template <bool g_> v_t<g_> v_t<g_>::g() const {
  static_assert(g_, "cannot call grad on non-tracked");
  if (!t_->at(i_).has_grad()) {
    return {};
  } else {
    return v_t<g_>::preexisting(t_, t_->at(i_).g_);
  }
}

template <bool g_> bool v_t<g_>::has_g() const {
  static_assert(g_, "cannot call grad on non-tracked");
  return t_->at(i_).has_grad();
}

// template <bool g> void v_t<g>::set_g(v_t const &r) { return t_->at(i_).g_ =
// r; }

template <bool g> auto v_t<g>::cs() const { return t_->at(i_).cs_; }

template <bool g> r_t re(v_t<g> const &v) { return v.t()->v_at(v.i()); }

template <typename mix> struct u_op : mix {
  using mix::app;
  using mix::bf;

  template <bool ga> auto operator()(v_t<ga> const &a) const {
    if constexpr (!ga) {
      return app(re(a));
    } else {
      idx_t idx = a.t()->emplace(tp_dt(app(re(a)), {a.i(), sntnl_idx}, bf()));
      return v_t<true>::preexisting(a.t(), idx);
    }
  }
};

void init_grad_if_not_set_at(tape_t *t, idx_t x, r_t r = {}) {
  if (!t->at(x).has_grad()) {
    auto g = v_t<true>{r, t};
    t->at(x).g_ = g.i();
  }
}

struct negmix {
  static void backward(tape_t *t, idx_t f_i);
  static r_t app(r_t a) { return -a; }
  static back_f bf() { return {"neg", backward}; }
};
template <typename T> auto neg(T const &a) { return u_op<negmix>()(a); }

r_t relu(r_t a) { return a > 0 ? a : 0; }

struct relumix {
  static void backward(tape_t *t, idx_t f_i);
  static r_t app(r_t a) { return relu(a); }
  static back_f bf() { return {"relu", backward}; }
};
template <typename T> auto relu(T const &a) { return u_op<relumix>()(a); }

struct sinmix {
  static void backward(tape_t *t, idx_t f_i);
  static r_t app(r_t a) { return std::sin(a); }
  static back_f bf() { return {"sin", backward}; }
};
template <typename T> auto sin(T const &a) { return u_op<sinmix>()(a); }

struct cosmix {
  static void backward(tape_t *t, idx_t f_i);
  static r_t app(r_t a) { return std::cos(a); }
  static back_f bf() { return {"cos", backward}; }
};
template <typename T> auto cos(T const &a) { return u_op<cosmix>()(a); }

struct tanhmix {
  static void backward(tape_t *t, idx_t f_i);
  static r_t app(r_t a) { return std::tanh(a); }
  static back_f bf() { return {"tanh", backward}; }
};
template <typename T> auto tanh(T const &a) { return u_op<tanhmix>()(a); }

struct identitymix {
  static void backward(tape_t *t, idx_t f_i);
  static r_t app(r_t a) { return a; }
  static back_f bf() { return {"identity", backward}; }
};
template <typename T> auto identity(T const &a) {
  return u_op<identitymix>()(a);
}

template <typename mix> struct b_op : mix {
  using mix::app;
  using mix::bf;

  template <bool ga, bool gb>
  v_t<true> make_ret(v_t<ga> const &a, v_t<gb> const &b) const {
    idx_t x = a.t()->emplace(tp_dt(app(re(a), re(b)), {a.i(), b.i()}, bf()));
    return v_t<true>::preexisting(a.t(), x);
  }

  template <bool ga, bool gb>
  auto operator()(v_t<ga> const &a, v_t<gb> const &b) const {
    if constexpr (!ga && !gb) {
      return app(re(a), re(b));
    } else {
      return make_ret(a, b);
    }
  }
  template <bool ga> auto operator()(v_t<ga> const &a, r_t const br) const {
    if constexpr (!ga) {
      return app(re(a), br);
    } else {
      v_t<false> b(br, a.t());
      return make_ret(a, b);
    }
  }
  template <bool gb> auto operator()(r_t const ar, v_t<gb> const &b) const {
    if constexpr (!gb) {
      return app(ar + re(b));
    } else {
      v_t<false> a(ar, b.t());
      return make_ret(a, b);
    }
  }
};

template <typename T> struct is_v_or_r_t : std::false_type {};
template <> struct is_v_or_r_t<v_t<true>> : std::true_type {};
template <> struct is_v_or_r_t<v_t<false>> : std::true_type {};
template <> struct is_v_or_r_t<r_t> : std::true_type {};

struct addmix {
  static void backward(tape_t *t, idx_t f_i);
  static r_t app(r_t a, r_t b) { return a + b; }
  static back_f bf() { return {"add", backward}; }
};
template <typename T, typename U> auto operator+(T const &a, U const &b) {
  return b_op<addmix>()(a, b);
}

struct submix {
  static void backward(tape_t *t, idx_t f_i);
  static r_t app(r_t a, r_t b) { return a - b; }
  static back_f bf() { return {"sub", backward}; }
};
template <typename T, typename U> auto operator-(T const &a, U const &b) {
  return b_op<submix>()(a, b);
}

struct mulmix {
  static void backward(tape_t *t, idx_t f_i);
  static r_t app(r_t a, r_t b) { return a * b; }
  static back_f bf() { return {"mul", backward}; }
};
template <typename T, typename U> auto operator*(T const &a, U const &b) {
  return b_op<mulmix>()(a, b);
}

struct divmix {
  static void backward(tape_t *t, idx_t f_i);
  static r_t app(r_t a, r_t b) { return a / b; }
  static back_f bf() { return {"div", backward}; }
};
template <typename T, typename U> auto operator/(T const &a, U const &b) {
  return b_op<divmix>()(a, b);
}

auto unary_op_backwards_data(tape_t *t, idx_t f_i) {
  v_t<true> f = v_t<true>::preexisting(t, f_i);
  auto const [x_i, _] = f.cs();
  init_grad_if_not_set_at(t, x_i);
  v_t<true> x = v_t<true>::preexisting(t, x_i);
  return std::make_tuple(f, x, x_i);
}

auto binary_op_backwards_data(tape_t *t, idx_t f_i) {
  v_t<true> f = v_t<true>::preexisting(t, f_i);
  auto const [x_i, y_i] = f.cs();
  // init_grad_if_not_set_at(t, x_i);
  // init_grad_if_not_set_at(t, y_i);
  v_t<true> x = v_t<true>::preexisting(t, x_i);
  v_t<true> y = v_t<true>::preexisting(t, y_i);
  return std::make_tuple(f, x, y, x_i, y_i);
}

void negmix::backward(tape_t *t, idx_t f_i) {
  auto const [f, x, x_i] = unary_op_backwards_data(t, f_i);
  auto g = x.g() + f.g() * -1.0;
  t->at(x_i).g_ = g.i();
}

void relumix::backward(tape_t *t, idx_t f_i) {
  auto const [f, x, x_i] = unary_op_backwards_data(t, f_i);
  auto g = x.g() + (re(x) > 0 ? f.g() : v_t<true>(0.0, t));
  t->at(x_i).g_ = g.i();
}

void sinmix::backward(tape_t *t, idx_t f_i) {
  auto const [f, x, x_i] = unary_op_backwards_data(t, f_i);
  auto g = x.g() + cos(x) * f.g();
  t->at(x_i).g_ = g.i();
}

void cosmix::backward(tape_t *t, idx_t f_i) {
  auto const [f, x, x_i] = unary_op_backwards_data(t, f_i);
  auto g = x.g() - sin(x) * f.g();
  t->at(x_i).g_ = g.i();
}

void tanhmix::backward(tape_t *t, idx_t f_i) {
  auto const [f, x, x_i] = unary_op_backwards_data(t, f_i);
  auto g = x.g() + (1.0 - (f * f)) * f.g();
  t->at(x_i).g_ = g.i();
}

void identitymix::backward(tape_t *t, idx_t f_i) {
  auto const [f, x, x_i] = unary_op_backwards_data(t, f_i);
  auto g = x.g() + f.g();
  t->at(x_i).g_ = g.i();
}

void addmix::backward(tape_t *t, idx_t f_i) {
  auto const [f, x, y, x_i, y_i] = binary_op_backwards_data(t, f_i);
  v_t<true> g_x;
  v_t<true> g_y;
  if (t->at(x_i).has_grad()) {
    g_x = x.g() + f.g();
  } else {
    g_x = f.g();
  }
  t->at(x_i).g_ = g_x.i();
  if (t->at(y_i).has_grad()) {
    g_y = y.g() + f.g();
  } else {
    g_y = f.g();
  }
  t->at(y_i).g_ = g_y.i();
}
void submix::backward(tape_t *t, idx_t f_i) {
  auto const [f, x, y, x_i, y_i] = binary_op_backwards_data(t, f_i);
  auto g_x = x.g() + f.g();
  t->at(x_i).g_ = g_x.i();
  auto g_y = y.g() - f.g();
  t->at(y_i).g_ = g_y.i();
}
void mulmix::backward(tape_t *t, idx_t f_i) {
  auto const [f, x, y, x_i, y_i] = binary_op_backwards_data(t, f_i);
  v_t<true> g_x;
  v_t<true> g_y;
  if (t->at(x_i).has_grad()) {
    g_x = x.g() + f.g() * y;
  } else {
    g_x = f.g() * y;
  }
  t->at(x_i).g_ = g_x.i();
  if (t->at(y_i).has_grad()) {
    g_y = y.g() + f.g() * x;
  } else {
    g_y = f.g() * x;
  }
  t->at(y_i).g_ = g_y.i();
}
void divmix::backward(tape_t *t, idx_t f_i) {
  auto const [f, x, y, x_i, y_i] = binary_op_backwards_data(t, f_i);
  auto g_x = x.g() + f.g() / y;
  t->at(x_i).g_ = g_x.i();
  auto g_y = y.g() - f.g() * x / (y * y);
  t->at(y_i).g_ = g_y.i();
}

inline std::string as_dot(tape_t &t, size_t start_count = 0);

template <bool b>
void backward(v_t<b> const &x, std::vector<uint8_t> &visited) {
  visited.clear();
  visited.resize(x.i() + 1, 0);
  tape_t *t = x.t();
  init_grad_if_not_set_at(t, x.i(), 1.0);

  stk<idx_t> st;
  st.push(x.i());

  // AKS_PRINT("********************\n" << as_dot(*t));

  while (!st.empty()) {
    // AKS_PRINT("///////////////////////\n" << as_dot(*t));
    const idx_t ci = st.top();
    st.pop();
    if (visited[ci] || t->at(ci).is_leaf())
      continue;
    visited[ci] = 1;
    init_grad_if_not_set_at(t, ci);
    t->at(ci).bf_.f_(t, ci);
    for (auto const &c : t->at(ci).cs_) {
      if (c != sntnl_idx) {
        st.push(c);
      }
    }
  }
}

} // namespace aks

#include <iomanip>
#include <sstream>

#include <unordered_map>

namespace aks {

template <bool g> tp_dt const &d(v_t<g> const &v) { return v.t()->at(v.i()); }

template <bool g> std::ostream &operator<<(std::ostream &os, v_t<g> const &v) {
  os << "v_t(";
  if (v.t()) {
    os << "val=";
    os << re(v);
    os << ", idx=";
    os << v.i();
  } else {
    os << "";
  }
  os << ")";
  return os;
}

std::string to_string(tp_dt const &v) {
  std::stringstream ss;
  ss << std::setprecision(15) << v.v_ << " : [" << v.g_ << "]";
  return ss.str();
}

template <bool g>
inline std::string as_dot(v_t<g> const &iv, size_t i, size_t cnt,
                          std::vector<std::vector<size_t>> const &vdata,
                          size_t start_cnt) {
  auto check = [](auto x) { return x != sntnl_idx; };

  std::stringstream ss;

  size_t idx = i + start_cnt;
  tp_dt const &v = d(iv);

  auto is_binary_op = [](std::string const &x) {
    return x == "add" || x == "sub" || x == "mul" || x == "div";
  };
  auto is_unary_op = [](std::string const &x) {
    return x == "neg" || x == "tanh" || x == "relu" || x == "sin" ||
           x == "cos" || x == "identity";
  };

  if (!v.is_leaf()) {
    ss << "\n"
       << idx << " [label = \"[" << idx << "] " << std::setprecision(15)
       << to_string(v) << "\", fontcolor=blue, color=blue];\n";
    if (is_binary_op(v.bf_.n_)) {
      if (check(v.cs_[0])) {
        ss << v.cs_[0] + start_cnt << "->" << v.bf_.n_ << cnt << "; ";
      }
      if (check(v.cs_[1])) {
        ss << v.cs_[1] + start_cnt << "->" << v.bf_.n_ << cnt << "; ";
      }
      if (check(v.cs_[0]) || check(v.cs_[1])) {
        ss << v.bf_.n_ << cnt << "->" << idx << "; ";
      }
    } else if (is_unary_op(v.bf_.n_)) {
      if (check(v.cs_[0])) {
        ss << v.cs_[0] + start_cnt << "->" << v.bf_.n_ << cnt << "; ";
      }
      if (check(v.cs_[0])) {
        ss << v.bf_.n_ << cnt << "->" << idx << "; ";
      }
    }
  } else {
    ss << "\n"
       << idx << " [label = \"[" << idx << "] " << std::setprecision(15)
       << to_string(v) << "\", fontcolor=darkgreen, color=darkgreen];\n";
  }

  return ss.str();
}

inline std::string as_dot(tape_t &t, size_t start_count) {
  auto const &[data, vdata] = t.get_data();

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
    std::string op = data[idx].bf_.n_ ? data[idx].bf_.n_ : "";
    if (!opcnt.contains(op)) {
      opcnt[op] = start_count;
    }
    size_t cnt = opcnt[op]++;
    std::string str =
        as_dot(v_t<false>::preexisting(&t, idx), idx, cnt, vdata, start_count);
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

template <typename T>
std::ostream &operator<<(std::ostream &o, std::vector<T> const &vs) {
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

} // namespace aks

#include <random>

namespace aks {
std::mt19937_64 rng;
std::uniform_real_distribution<double> unif(-1.0, 1.0);

struct neuron {
  neuron(tape_t *tape, size_t nin, bool nonlin = true)
      : b(v_t<true>(unif(rng), tape)) {
    w.reserve(nin);
    for (size_t i = 0; i < nin; ++i) {
      w.push_back(v_t<true>(unif(rng) * std::sqrt(1.0 / double(nin)), tape));
    }
    nlin = nonlin;
  }

  v_t<true> operator()(std::vector<v_t<true>> const &xs) {
    v_t<true> result(0.0, b.t());
    result = result + b;
    for (size_t i = 0; i < w.size(); ++i) {
      result = result + (w[i] * xs[i]);
    }
    // v_t result = b + dot(w, xs);
    return nlin ? relu(result) : result;
  }

  std::vector<v_t<true>> parameters() {
    std::vector<v_t<true>> ret = w;
    ret.push_back(b);
    return ret;
  }

  v_t<true> b;
  std::vector<v_t<true>> w;
  bool nlin;
};

struct layer {
  layer(tape_t *tape, size_t nin, size_t nout, bool nonlin = true) {

    for (size_t i = 0; i < nout; ++i) {
      neurons.emplace_back(tape, nin, nonlin);
    }
  }

  void operator()(std::vector<v_t<true>> const &xs,
                  std::vector<v_t<true>> &activations) {
    activations.resize(neurons.size());
    for (size_t i = 0; i < neurons.size(); ++i) {
      activations[i] = neurons[i](xs);
    }
  }

  std::vector<v_t<true>> parameters() {
    std::vector<v_t<true>> ret;
    for (auto n : neurons) {
      auto p = n.parameters();
      ret.insert(ret.end(), p.begin(), p.end());
    }
    return ret;
  }

  std::vector<neuron> neurons;
};

struct mlp {
  mlp(tape_t *tape, size_t nin, std::vector<size_t> nouts) : tape_(tape) {
    std::vector<size_t> sz(1, nin);
    sz.insert(sz.end(), nouts.begin(), nouts.end());
    for (size_t i = 0; i < nouts.size(); ++i) {
      auto nin = sz[i];
      auto nout = sz[i + 1];
      auto nlin = (i != nouts.size() - 1);
      layers.emplace_back(tape, nin, nout, nlin);
    }
    last_activations.resize(layers.size());
  }

  std::vector<v_t<true>> const &operator()(std::vector<v_t<true>> const &xs) {
    for (auto &a : last_activations) {
      a.clear();
    }
    layers[0](xs, last_activations[0]);
    for (size_t i = 1; i < layers.size(); ++i) {
      layers[i](last_activations[i - 1], last_activations[i]);
    }
    return last_activations.back();
  }

  std::vector<v_t<true>> parameters() {
    std::vector<v_t<true>> ret;
    for (auto layer : layers) {
      auto p = layer.parameters();
      ret.insert(ret.end(), p.begin(), p.end());
    }
    return ret;
  }

  tape_t *tape_;
  std::vector<layer> layers;
  std::vector<std::vector<v_t<true>>> last_activations;
};

} // namespace aks

namespace aks {

template <typename X, typename U, typename F>
auto transform_in_place(U &&us, F &&func, std::vector<X> &ret) {
  ret.reserve(us.size());
  for (auto &u : us) {
    ret.emplace_back(func(u));
  }
}

template <typename X, typename T, typename U, typename F>
auto transform_in_place(T &&ts, U &&us, F &&func, std::vector<X> &ret) {
  assert(ts.size() == us.size());
  ret.reserve(us.size());
  for (size_t i = 0; i < us.size(); ++i) {
    auto &u = us[i];
    auto &t = ts[i];
    ret.emplace_back(func(t, u));
  }
}

template <typename X, typename T, typename U, typename V, typename F>
auto transform_in_place(T &&ts, U &&us, V &&vs, F &&func, std::vector<X> &ret) {
  assert(ts.size() == us.size());
  assert(us.size() == vs.size());
  ret.reserve(us.size());
  for (size_t i = 0; i < us.size(); ++i) {
    auto &u = us[i];
    auto &t = ts[i];
    auto &v = vs[i];
    ret.emplace_back(func(t, u, v));
  }
}

template <typename U, typename F> auto transform(U &&us, F &&func) {
  using out_t = decltype(func(us.front()));
  std::vector<out_t> ret;
  ret.reserve(us.size());
  for (auto &u : us) {
    ret.emplace_back(func(u));
  }
  return ret;
}

template <typename T, typename U, typename F>
auto transform(T &&ts, U &&us, F &&func) {
  using out_t = decltype(func(ts.front(), us.front()));
  assert(ts.size() == us.size());
  std::vector<out_t> ret;
  ret.reserve(us.size());
  for (size_t i = 0; i < us.size(); ++i) {
    auto &u = us[i];
    auto &t = ts[i];
    ret.emplace_back(func(t, u));
  }
  return ret;
}

template <typename T, typename U, typename V, typename F>
auto transform(T &&ts, U &&us, V &&vs, F &&func) {
  using out_t = decltype(func(ts.front(), us.front(), vs.front()));
  assert(ts.size() == us.size());
  assert(us.size() == vs.size());
  std::vector<out_t> ret;
  ret.reserve(us.size());
  for (size_t i = 0; i < us.size(); ++i) {
    auto &u = us[i];
    auto &t = ts[i];
    auto &v = vs[i];
    ret.emplace_back(func(t, u, v));
  }
  return ret;
}

} // namespace aks

int main() {
  using namespace aks;
  if (false) {
    tape_t t;
    v_t<true> x(0.5, &t);
    v_t<true> f = tanh(x);
    AKS_PRINT(f << ":" << x);
    std::vector<uint8_t> visited;
    backward(f, visited);
    AKS_PRINT(x.g());
    // AKS_PRINT(as_dot(t));
  }
  if (false) {
    auto func = [](auto x) {
      using namespace aks;
      using namespace std;
      return sin(x);
    };

    tape_t tape;
    mlp n(&tape, 1, {8, 8, 1});
    auto params = n.parameters();

    r_t learning = 0.01;
    size_t count = 0;
    size_t const max_iterations = 200000;

    auto linspace = [&](r_t a, r_t b, size_t n) {
      reals_t ret(n);
      for (size_t i = 0; i < n; ++i) {
        ret[i] = a + i * (b - a) / (n - 1);
      }
      return ret;
    };

    reals_t xs_ = linspace(-3.14, 3.14, 200);
    std::vector<std::vector<v_t<true>>> xs = transform(xs_, [&](r_t x) {
      return std::vector<v_t<true>>{v_t<true>(x, &tape)};
    });

    reals_t exact = transform(xs_, func);

    auto loss_func = [&](reals_t const &y, std::vector<v_t<true>> const &pred) {
      v_t<true> ret(0.0, &tape);

      for (size_t i = 0; i < y.size(); ++i) {
        v_t<true> j = (pred[i] - y[i]);
        ret = ret + (j * j);
      }

      return ret / r_t(y.size());
    };

    vec<v_t<true>> ypred;
    ypred.reserve(xs.size());
    vec<v_t<true>> grad_vals;
    grad_vals.reserve(params.size());
    std::vector<uint8_t> visited;
    while (count++ < max_iterations) {
      tape.push_store_point();

      transform_in_place(
          xs, [&](auto const &x) { return n(x)[0]; }, ypred);

      v_t<true> loss = loss_func(exact, ypred);
      // losses.push_back(as_real(loss));

      // AKS_PRINT(as_dot(tape));

      if (count == 1 || count % 500 == 0) {
        AKS_PRINT_AS("", count << ":" << loss << ":" << learning);
      }

      // backwards_tree(tape, loss, grads, visited);
      backward(loss, visited);

      // AKS_PRINT(as_dot(tape));
      // return 0;

      grad_vals = transform(params, [](v_t<true> v) { return v.g(); });

      if (count % 2500 == 0) {
        if (learning > 0.0001) {
          learning *= 0.99;
        }
      }

      for (size_t i = 0; i < params.size(); ++i) {
        auto &p = params[i];
        auto &grad = grad_vals[i];

        p = re(p) - learning * grad;
      }

      tape.pop_store_point();
    }

    std::vector<r_t> x_test = linspace(-3.14, 3.14, 50);

    std::vector<std::vector<v_t<true>>> xs_test = transform(x_test, [&](r_t x) {
      return std::vector<v_t<true>>{v_t<true>(x, &tape)};
    });

    std::vector<r_t> result = transform(
        xs_test, [&](std::vector<v_t<true>> const &x) { return re(n(x)[0]); });

    std::vector<r_t> test_exact = transform(x_test, func);

    AKS_PRINT(test_exact);
    AKS_PRINT(result);

    //
    //
    //  AKS_PRINT(losses);
    // AKS_PRINT(en(std::vector<real>{xs_[0]}));
    // AKS_PRINT(en(std::vector<real>{xs_[1]}));
  }
  if (true) {
    tape_t t;
    v_t<true> x(3, &t);
    v_t<true> y(5, &t);
    // auto y_sq = y * y;
    // auto y_sq_x = y_sq * x;
    v_t<true> f = (x * x) * (y * y); // y_sq_x + y_sq_x;
    std::vector<uint8_t> visited;

    AKS_PRINT(f);
    // AKS_PRINT("<<<<<<<<<<<<<< 0 >>>>>>>>>>>>>>>\n" << as_dot(t));

    t.clear_grads(f);
    backward(f, visited);
    AKS_PRINT(y.g());
    AKS_PRINT(x.g());
    // AKS_PRINT("<<<<<<<<<<<<<< 1 >>>>>>>>>>>>>>>\n" << as_dot(t));
    auto dfdx = x.g();
    t.clear_grads(f);
    backward(dfdx, visited);
    AKS_PRINT(y.g());
    AKS_PRINT(x.g());
    // AKS_PRINT("<<<<<<<<<<<<<< 2 >>>>>>>>>>>>>>>\n" << as_dot(t));
    auto d2fdx2 = x.g();
    t.clear_grads(f);
    backward(d2fdx2, visited);
    AKS_PRINT(x.g());
    AKS_PRINT(y.g());
    AKS_PRINT("<<<<<<<<<<<<<< 3 >>>>>>>>>>>>>>>\n" << as_dot(t));
    if (x.has_g()) {
      auto d3fdx3 = x.g();
      t.clear_grads(f);
      backward(d3fdx3, visited);
      AKS_PRINT(x.g());
    }
    if (y.has_g()) {
      auto d3fdx2dy = y.g();
      t.clear_grads(f);
      backward(d3fdx2dy, visited);
      AKS_PRINT(y.g());
    }
    // AKS_PRINT("<<<<<<<<<<<<<< 4 >>>>>>>>>>>>>>>\n" << as_dot(t));
    // AKS_PRINT(as_dot(t));
  }
  if (false) {
    tape_t t;
    v_t<true> x(3.0, &t);
    v_t<true> f = x * x;
    std::vector<uint8_t> visited;
    AKS_PRINT(f);
    t.clear_grads(f);
    backward(f, visited);
    AKS_PRINT(x.g());
    // AKS_PRINT(as_dot(t));
    backward(x.g(), visited);
    // AKS_PRINT(as_dot(t));
    AKS_PRINT(x.g());
  }
  if (false) {
    tape_t t;
    // t.reserve(200);
    v_t<true> a(3.0, &t);
    v_t<true> b(6.0, &t);
    v_t<true> c = (a + b) * 2.0;
    c = (c * c) / a;
    c = neg(c - b);
    std::vector<uint8_t> visited;
    backward(c, visited);
    AKS_PRINT(a.g());
    backward(a.g(), visited);
    AKS_PRINT(as_dot(t));
    AKS_PRINT(a.g());
  }
  {
    tape_t t;
    v_t<true> a(3.0, &t);
    v_t<false> b(6.0, &t);
    auto c = a + b;
  }
  {
    tape_t t;
    v_t<false> a(3.0, &t);
    v_t<false> b(6.0, &t);
    auto c = a + b;
  }
  {
    tape_t t;
    v_t<true> a(3.0, &t);
    auto c = a + 6.0;
  }
  {
    tape_t t;
    v_t<false> a(3.0, &t);
    auto c = a + 6.0;
  }

  return 0;
}

#endif
