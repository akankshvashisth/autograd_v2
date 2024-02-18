
#define AKS_NO_VCL

#include "autograd_v2.hpp"

#include <optional>
#include <random>

namespace {

using double_t = double;
using float_t = float;

template <typename value_type> struct autograd_traits {
  using value_t = value_type;
  using var_t = aks::var_t<value_type>;
  using tape_t = aks::tape_t<value_type>;
  using tape_context_t = aks::tape_context<value_type>;
  using vec_var_t = aks::vec_t<var_t>;
  using vec_value_t = aks::vec_t<value_t>;
  using vec_vec_var_t = aks::vec_t<vec_var_t>;
  using vec_vec_value_t = aks::vec_t<vec_value_t>;
};

using ag_d = autograd_traits<double_t>;
// using ag_d = autograd_traits<vcl::Vec2d>;
using ag_f = autograd_traits<float_t>;

constexpr bool QUIET_PASS = true;
constexpr bool QUIET_FAIL = false;
constexpr bool ASSERT_FAIL = false;
static size_t TOTAL_TEST_RUN = 0;
static size_t TOTAL_TEST_PASS = 0;
static size_t TOTAL_TEST_FAIL = 0;

#define AKS_PRINT(EXPR)                                                        \
  using namespace aks::vcl_detail;                                             \
  std::cout << std::setprecision(15) << #EXPR << " = " << EXPR << std::endl

#define AKS_PRINT_AS(NAME, EXPR)                                               \
  std::cout << std::setprecision(15) << NAME << " " << EXPR << "       \t("    \
            << #EXPR << ")" << std::endl

#define AKS_CHECK_PRINT(EXPR, EXPR_VAL, EXPECTED)                              \
  do {                                                                         \
    using namespace aks::vcl_detail;                                           \
    ++TOTAL_TEST_RUN;                                                          \
    const double_t diff__ =                                                    \
        std::abs(aks::vcl_detail::get_first_as<double>(EXPR_VAL) -             \
                 aks::vcl_detail::get_first_as<double>(EXPECTED));             \
    if (std::isnan(aks::vcl_detail::get_first_as<double>(EXPR_VAL)) ||         \
        diff__ > 1e-4) {                                                       \
      ++TOTAL_TEST_FAIL;                                                       \
      if (!QUIET_FAIL) {                                                       \
        std::cout << std::setprecision(18) << "\nCHECK FAILED: " << #EXPR      \
                  << " = " << EXPR_VAL << " != " << EXPECTED << " (" << diff__ \
                  << ")"                                                       \
                  << " on line " << __LINE__ << " in " << __FILE__             \
                  << std::endl;                                                \
      } else {                                                                 \
        std::cout << "+";                                                      \
      }                                                                        \
      assert(!ASSERT_FAIL);                                                    \
    } else {                                                                   \
      ++TOTAL_TEST_PASS;                                                       \
      if (!QUIET_PASS) {                                                       \
        std::cout << std::setprecision(15) << "____pass____: " << #EXPR        \
                  << " = " << aks::vcl_detail::get_first_as<double>(EXPR_VAL)  \
                  << std::endl;                                                \
      } else {                                                                 \
        std::cout << ".";                                                      \
      }                                                                        \
    }                                                                          \
  } while (false)

#define AKS_CHECK_VARIABLE(EXPR, EXPECTED)                                     \
  AKS_CHECK_PRINT(EXPR, EXPR.value(), EXPECTED)

#define AKS_CHECK_VALUE(EXPR, EXPECTED) AKS_CHECK_PRINT(EXPR, (EXPR), EXPECTED)

#define AKS_CHECK_TRUE(EXPR) AKS_CHECK_PRINT(EXPR, (EXPR), (1))

#define AKS_CHECK_FALSE(EXPR) AKS_CHECK_PRINT(EXPR, (EXPR), (0))

#define AKS_CHECK_EQUAL(A, B) AKS_CHECK_PRINT(A, (A), (B))

#define AKS_CHECK_VALUES(RESULT, EXPECTED)                                     \
  do {                                                                         \
    AKS_CHECK_TRUE(RESULT.size() == EXPECTED.size());                          \
    for (size_t i = 0; i < RESULT.size(); ++i) {                               \
      AKS_CHECK_VALUE(RESULT[i], EXPECTED[i]);                                 \
    }                                                                          \
  } while (false)

#define AKS_CHECK_VARIABLES(RESULT, EXPECTED)                                  \
  do {                                                                         \
    AKS_CHECK_TRUE(RESULT.size() == EXPECTED.size());                          \
    for (size_t i = 0; i < RESULT.size(); ++i) {                               \
      AKS_CHECK_VALUE(RESULT[i].value(), EXPECTED[i]);                         \
    }                                                                          \
  } while (false)

} // namespace

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

namespace {
using ag_mlp = autograd_traits<double_t>;

struct neuron {
  neuron(std::mt19937_64 &rng, ag_mlp::tape_t *t, size_t n_in, bool nlin)
      : tape(t), nonlinearity(nlin) {

    const ag_mlp::value_t k = ag_mlp::value_t{1.0} / ag_mlp::value_t(n_in);
    const ag_mlp::value_t sqrt_k = std::sqrt(k);
    std::uniform_real_distribution<ag_mlp::value_t> unif(-k, k);

    ws.reserve(n_in);

    b = tape->new_variable(unif(rng), true);
    for (size_t i = 0; i < n_in; i++) {
      ws.emplace_back(tape->new_variable(unif(rng), true));
    }

    params = parameters_impl();
  }

  void forward(ag_mlp::vec_var_t const &xs, ag_mlp::var_t &out) {
    ag_mlp::var_t r = b + dot(ws, xs);
    out = (nonlinearity ? relu(r) : r);
  }

  ag_mlp::vec_var_t const &parameters() { return params; }

  ag_mlp::vec_var_t parameters_impl() {
    ag_mlp::vec_var_t ret;
    ret.reserve(ws.size() + 1);
    ret.push_back(b);
    ret.insert(ret.end(), ws.begin(), ws.end());
    return ret;
  }

  ag_mlp::tape_t *tape;
  bool nonlinearity = true;
  ag_mlp::var_t b;
  ag_mlp::vec_var_t ws;
  ag_mlp::vec_var_t params;
};

struct layer_linear {
  layer_linear(std::mt19937_64 &rng, ag_mlp::tape_t *t, size_t n_in,
               size_t n_out, bool nlin) {
    for (size_t i = 0; i < n_out; i++) {
      neurons.emplace_back(rng, t, n_in, nlin);
    }
    params = parameters_impl();
  }

  void forward(ag_mlp::vec_var_t const &x, ag_mlp::vec_var_t &out) {
    if (out.size() != neurons.size()) {
      out.resize(neurons.size());
    }
    for (size_t i = 0; i < neurons.size(); i++) {
      neurons[i].forward(x, out[i]);
    }
  }

  ag_mlp::vec_var_t const &parameters() { return params; }

private:
  ag_mlp::vec_var_t parameters_impl() {
    ag_mlp::vec_var_t ret;
    for (auto &n : neurons) {
      ag_mlp::vec_var_t const &p = n.parameters();
      ret.insert(ret.end(), p.begin(), p.end());
    }
    return ret;
  }

  aks::vec_t<neuron> neurons;
  ag_mlp::vec_var_t params;

  friend struct layer_linear_inf;
};

struct mlp {

  mlp(size_t n_in, aks::vec_t<size_t> n_outs, unsigned long long seed = 42) {
    rng.seed(seed);
    aks::vec_t<size_t> sz(1, n_in);
    sz.insert(sz.end(), n_outs.begin(), n_outs.end());

    for (size_t i = 0; i < n_outs.size(); i++) {
      layers.emplace_back(rng, &tape, sz[i], sz[i + 1],
                          (i != n_outs.size() - 1));
    }

    params = parameters_impl();
  }

  void forward(ag_mlp::vec_var_t const &x, ag_mlp::vec_var_t &out,
               ag_mlp::vec_var_t &buffer) {
    out.clear();
    out.insert(out.end(), x.begin(), x.end());
    for (auto &layer : layers) {
      buffer.clear();
      layer.forward(out, buffer);
      buffer.swap(out);
    }
    // out has the result
  }

  ag_mlp::vec_var_t const &parameters() { return params; }

  ag_mlp::vec_var_t parameters_impl() {
    ag_mlp::vec_var_t ret;
    for (auto &l : layers) {
      ret.insert(ret.end(), l.parameters().begin(), l.parameters().end());
    }
    return ret;
  }

  ag_mlp::tape_t tape;
  aks::vec_t<layer_linear> layers;
  ag_mlp::vec_var_t params;
  ag_mlp::var_t last_neural_node;
  std::mt19937_64 rng;
};

struct neuron_inf {
  neuron_inf(neuron const &n) {
    nonlinearity = n.nonlinearity;
    b = n.b.value();
    ws.reserve(n.ws.size());
    for (const auto &w : n.ws) {
      ws.push_back(w.value());
    }
  }

  void forward(ag_mlp::vec_value_t const &xs, ag_mlp::value_t &out) {
    out = b;
    for (size_t i = 0; i < xs.size(); ++i) {
      out += xs[i] * ws[i];
    }
    if (nonlinearity) {
      out = out > ag_mlp::value_t(0.0) ? out : ag_mlp::value_t(0.0);
    }
  }

  bool nonlinearity = true;
  ag_mlp::value_t b;
  ag_mlp::vec_value_t ws;
};

struct layer_linear_inf {
  layer_linear_inf(layer_linear const &ll) {
    neurons.reserve(ll.neurons.size());
    for (auto const &n : ll.neurons) {
      neurons.emplace_back(n);
    }
  }

  void forward(ag_mlp::vec_value_t const &x, ag_mlp::vec_value_t &out) {
    if (out.size() != neurons.size()) {
      out.resize(neurons.size());
    }
    for (size_t i = 0; i < neurons.size(); i++) {
      neurons[i].forward(x, out[i]);
    }
  }

private:
  aks::vec_t<neuron_inf> neurons;
};

struct mlp_inf {
  mlp_inf(mlp const &m) {
    layers.reserve(m.layers.size());
    for (auto const &lay : m.layers) {
      layers.emplace_back(lay);
    }
  }

  void forward(ag_mlp::vec_value_t const &x, ag_mlp::vec_value_t &out,
               ag_mlp::vec_value_t &buffer) {
    out.clear();
    out.insert(out.end(), x.begin(), x.end());
    for (auto &layer : layers) {
      buffer.clear();
      layer.forward(out, buffer);
      buffer.swap(out);
    }
    // out has the result
  }

  ag_mlp::vec_value_t forward(ag_mlp::vec_value_t const &x) {
    ag_mlp::vec_value_t out;
    ag_mlp::vec_value_t buffer;
    out.clear();
    out.insert(out.end(), x.begin(), x.end());
    for (auto &layer : layers) {
      buffer.clear();
      layer.forward(out, buffer);
      buffer.swap(out);
    }
    return out;
  }

  aks::vec_t<layer_linear_inf> layers;
};

struct optimizer {
  optimizer(ag_mlp::tape_t *t, ag_mlp::vec_var_t params, ag_mlp::value_t lr)
      : tape_(t), learning_rate_(lr), params_(std::move(params)) {
    for (auto const &p : params_) {
      assert(p.is_alive());
      assert(p.n().requires_grad_);
    }
  }

  void zero_grad() { tape_->zero_grad(); }

  void step(std::optional<ag_mlp::value_t> maybe_lr = std::nullopt) {
    ag_mlp::value_t lr = maybe_lr.value_or(learning_rate_);
    for (auto &p : params_) {
      ag_mlp::var_t g = grad(p);
      ag_mlp::value_t update = p.value() - (lr * g.value());
      p.update_in_place(update);
    }
  }

  ag_mlp::tape_t *tape_;
  ag_mlp::value_t learning_rate_;
  ag_mlp::vec_var_t params_;
};

template <typename real_t>
aks::vec_t<real_t> linspace(real_t start, real_t end, size_t num,
                            bool endpoint = true) {
  aks::vec_t<real_t> ret(num);
  real_t step = (end - start) / (num - (endpoint ? 1 : 0));
  for (size_t i = 0; i < num; ++i) {
    ret[i] = start + step * i;
  }
  return ret;
}

template <typename real_t>
aks::vec_t<aks::var_t<real_t>> put_on_tape(aks::tape_t<real_t> *tape,
                                           aks::vec_t<real_t> const &xs,
                                           bool requires_grad = false) {
  aks::vec_t<aks::var_t<real_t>> ret;
  ret.reserve(xs.size());
  for (auto const &x : xs) {
    ret.push_back(tape->new_variable(x, requires_grad));
  }
  return ret;
}

template <typename real_t>
aks::vec_t<real_t> get_values(aks::vec_t<aks::var_t<real_t>> const &xs) {
  aks::vec_t<real_t> ret;
  ret.reserve(xs.size());
  for (auto const &x : xs) {
    assert(x.is_alive());
    ret.push_back(x.value());
  }
  return ret;
}

template <typename real_t>
aks::vec_t<aks::var_t<real_t>>
get_grads(aks::vec_t<aks::var_t<real_t>> const &xs) {
  aks::vec_t<aks::var_t<real_t>> ret;
  ret.reserve(xs.size());
  for (auto const &x : xs) {
    assert(x.is_alive());
    assert(x.requires_grad());
    ret.push_back(aks::grad(x));
  }
  return ret;
}

template <typename real_t>
aks::map_t<aks::var_t<real_t>, aks::var_t<real_t>>
get_grads_map(aks::vec_t<aks::var_t<real_t>> const &xs) {
  aks::map_t<aks::var_t<real_t>, aks::var_t<real_t>> ret;
  ret.reserve(xs.size());
  for (auto const &x : xs) {
    assert(x.is_alive());
    assert(x.requires_grad());
    ret[x] = aks::grad(x);
  }
  return ret;
}

template <typename real_t, typename... var_ts>
auto get_grads(aks::var_t<real_t> x, var_ts... xs) {
  return std::make_tuple(aks::grad(x), aks::grad(xs)...);
}

template <typename real_t> void build_gradients(aks::var_t<real_t> f) {
  assert(f.is_alive());
  assert(f.requires_grad());
  f.t().zero_grad();
  aks::backward(f);
}

template <typename real_t>
aks::var_t<real_t> gradient(aks::var_t<real_t> f, aks::var_t<real_t> x) {
  build_gradients(f);
  return aks::grad(x);
}

template <typename real_t, typename... vars>
auto gradient(aks::var_t<real_t> f, aks::var_t<real_t> x, vars... xs) {
  build_gradients(f);
  return get_grads(x, xs...);
}

template <typename real_t>
aks::vec_t<aks::var_t<real_t>>
gradient(aks::var_t<real_t> f, aks::vec_t<aks::var_t<real_t>> const &xs) {
  build_gradients(f);
  return get_grads(xs);
}

template <typename real_t>
aks::vec_t<aks::vec_t<aks::var_t<real_t>>>
gradient(aks::var_t<real_t> f,
         aks::vec_t<aks::vec_t<aks::var_t<real_t>>> const &xss) {
  build_gradients(f);
  aks::vec_t<aks::vec_t<aks::var_t<real_t>>> ret;
  ret.reserve(xss.size());
  for (auto const &xs : xss) {
    ret.emplace_back(get_grads(xs));
  }
  return ret;
}

template <typename real_t>
aks::vec_t<aks::var_t<real_t>>
gradient(aks::vec_t<aks::var_t<real_t>> const &fs,
         aks::vec_t<aks::var_t<real_t>> const &xs) {
  aks::vec_t<aks::var_t<real_t>> ret;
  ret.reserve(fs.size());
  assert(fs.size() == xs.size());
  for (size_t i = 0; i < fs.size(); ++i) {
    ret.push_back(gradient(fs[i], xs[i]));
  }
  return ret;
}

template <typename real_t>
aks::map_t<aks::var_t<real_t>, aks::var_t<real_t>>
gradient_map(aks::var_t<real_t> f, aks::vec_t<aks::var_t<real_t>> const &xs) {
  build_gradients(f);
  return get_grads_map(xs);
}

template <typename real_t>
aks::map_t<aks::var_t<real_t>, aks::var_t<real_t>>
gradient_map(aks::vec_t<aks::var_t<real_t>> const &fs,
             aks::vec_t<aks::var_t<real_t>> const &xs) {
  aks::map_t<aks::var_t<real_t>, aks::var_t<real_t>> ret;

  assert(fs.size() == xs.size());
  for (size_t i = 0; i < fs.size(); ++i) {
    ret[xs[i]] = gradient(fs[i], xs[i]);
  }
  return ret;
}
} // namespace

namespace {

void test_01() {
  std::cout << "\ntest_01" << std::endl;
  using namespace aks;

  ag_d::tape_t t;

  const ag_d::var_t x = t.new_variable(3.0, true);
  ag_d::var_t y = t.new_variable(5.0, true);

  {
    ag_d::tape_context_t ctxt(t);

    ag_d::var_t f = (x * x * x * x * x * x * x * x);
    AKS_CHECK_VARIABLE(x, 3);
    AKS_CHECK_VARIABLE(y, 5);
    AKS_CHECK_VARIABLE(f, 6561);
    vec_t<ag_d::value_t> expected = {17496,  40824,  81648, 136080, 181440,
                                     181440, 120960, 40320, 0};

    for (int i = 0; i < 9; ++i) {
      t.zero_grad();
      backward(f);
      f = grad(x);
      AKS_CHECK_VARIABLE(f, expected[i]);
    }
    AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 11717);
    AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 2);
  }

  {
    ag_d::tape_context_t ctxt(t);
    ag_d::var_t f = (y * y) * (x * x) * (y * y * y);

    vec_t<ag_d::value_t> expected = {28125, 22500, 13500, 5400, 1080,
                                     0,     0,     0,     0};
    AKS_CHECK_VARIABLE(f, 28125);
    for (int i = 0; i < 9; ++i) {
      t.zero_grad();
      backward(f);
      f = grad(y);
      AKS_CHECK_VARIABLE(f, expected[i]);
    }
    AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 1038);
    AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 2);
  }

  {
    ag_d::tape_context_t ctxt(t);

    ag_d::var_t f = (x * x) / (y * y);
    vec_t<ag_d::value_t> expected = {-0.144, 0.0864, -0.06912, 0.06912,
                                     -0.0829439999999999};
    AKS_CHECK_VARIABLE(f, 0.36);

    for (int i = 0; i < 5; ++i) {
      t.zero_grad();
      backward(f);
      f = grad(y);
      AKS_CHECK_VARIABLE(f, expected[i]);
    }
    AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 766);
    AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 154);
  }

  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 2);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 0);
}

void test_02() {
  std::cout << "\ntest_02" << std::endl;
  using namespace aks;
  ag_d::tape_t t;
  const ag_d::var_t x = t.new_variable(3.0, true);
  const ag_d::var_t y = t.new_variable(8.0);

  ag_d::var_t f = pow(x, y);

  t.push_state();
  AKS_CHECK_VARIABLE(x, 3);
  AKS_CHECK_VARIABLE(y, 8);
  AKS_CHECK_VARIABLE(f, 6561);
  vec_t<ag_d::value_t> expected = {17496,  40824,  81648, 136080, 181440,
                                   181440, 120960, 40320, 0};

  for (int i = 0; i < 9; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(x);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 73);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 4);
  t.pop_state();
}

void test_02_f() {
  std::cout << "\ntest_02_f" << std::endl;
  using namespace aks;
  using ag = ag_f;
  ag::tape_t t;
  const ag::var_t x = t.new_variable(3.0, true);
  const ag::var_t y = t.new_variable(8.0);

  ag::var_t f = pow(x, y);

  t.push_state();
  AKS_CHECK_VARIABLE(x, 3);
  AKS_CHECK_VARIABLE(y, 8);
  AKS_CHECK_VARIABLE(f, 6561);
  vec_t<ag_d::value_t> expected = {17496,  40824,  81648, 136080, 181440,
                                   181440, 120960, 40320, 0};

  for (int i = 0; i < 9; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(x);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 73);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 4);
  t.pop_state();
}

void test_03() {
  std::cout << "\ntest_03" << std::endl;
  using namespace aks;
  ag_d::tape_t t;
  const ag_d::var_t x = t.new_variable(3.0, true);

  ag_d::var_t f = pow(x, ag_d::value_t(8.0));

  t.push_state();
  AKS_CHECK_VARIABLE(x, 3);
  AKS_CHECK_VARIABLE(f, 6561);
  vec_t<ag_d::value_t> expected = {17496,  40824,  81648, 136080, 181440,
                                   181440, 120960, 40320, 0};

  for (int i = 0; i < 9; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(x);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 73);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 4);
  t.pop_state();
}

void test_04() {
  std::cout << "\ntest_04" << std::endl;
  using namespace aks;
  ag_d::tape_t t;
  const ag_d::var_t x = t.new_variable(3.0, true);

  ag_d::var_t f = exp(log(exp(log(exp(log(x))))));

  t.push_state();
  AKS_CHECK_VARIABLE(x, 3);
  AKS_CHECK_VARIABLE(f, 3);
  vec_t<ag_d::value_t> expected = {1, 0, 0};

  for (int i = 0; i < 3; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(x);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 184);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 41);
  t.pop_state();
}

void test_05() {
  std::cout << "\ntest_05" << std::endl;
  using namespace aks;
  ag_d::tape_t t;
  const ag_d::var_t x = t.new_variable(1, true);

  ag_d::var_t f = exp(exp(x));

  t.push_state();
  AKS_CHECK_VARIABLE(x, 1);
  AKS_CHECK_VARIABLE(f, 15.1542622415);
  vec_t<ag_d::value_t> expected = {
      41.1935556747, 153.169249515,     681.502130990,
      3478.70705883, 19853.4050763,     124537.473663,
      848181.148608, 6213971.481006121, 48615295.31226263};

  for (int i = 0; i < 9; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(x);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 16669);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 4081);
  t.pop_state();
}

void test_06_01() {
  std::cout << "\ntest_06_01" << std::endl;
  using namespace aks;
  ag_d::tape_t t;
  const ag_d::var_t x =
      t.new_variable(std::numbers::pi_v<double_t> / ag_d::value_t(2.0), true);

  ag_d::var_t f = sin(x);

  t.push_state();
  AKS_CHECK_VARIABLE(x, std::numbers::pi_v<double_t> / 2.0);
  AKS_CHECK_VARIABLE(f, 1);
  vec_t<ag_d::value_t> expected = {0, -1, 0, 1, 0, -1, 0, 1, 0};

  for (int i = 0; i < 9; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(x);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 49);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 4);
  t.pop_state();
}

void test_06_02() {
  std::cout << "\ntest_06_02" << std::endl;
  using namespace aks;
  ag_d::tape_t t;
  const ag_d::var_t x =
      t.new_variable(std::numbers::pi_v<double_t> / ag_d::value_t(2.0), true);

  ag_d::var_t f = cos(x);

  t.push_state();
  AKS_CHECK_VARIABLE(x, std::numbers::pi_v<double_t> / ag_d::value_t(2.0));
  AKS_CHECK_VARIABLE(f, 0);
  vec_t<ag_d::value_t> expected = {-1, 0, 1, 0, -1, 0, 1, 0, -1};

  for (int i = 0; i < 9; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(x);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 51);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 3);
  t.pop_state();
}

void test_07() {
  std::cout << "\ntest_07" << std::endl;
  using namespace aks;
  ag_d::tape_t t;
  const ag_d::var_t x = t.new_variable(2, true);

  ag_d::var_t f = ((sin(x) ^ ag_d::value_t(2.0)) /
                   (log(x + ag_d::value_t(50.0)) ^ ag_d::value_t(2.0))) *
                      (ag_d::value_t(1.0) - (exp(x) ^ (-x))) +
                  cos(x);

  t.push_state();
  AKS_CHECK_VARIABLE(x, 2);
  AKS_CHECK_VARIABLE(f, -0.364157274408385);
  vec_t<ag_d::value_t> expected = {
      -0.953510374699207, 0.314131605670404,  1.160290265129676,
      -0.127327515031308, -2.350487124618407, 3.030458853334027,
      -1.213075727292083, -63.66425006790054, 463.4642048661533};

  for (int i = 0; i < 9; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(x);
    // AKS_PRINT(f);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 948883);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 214781);
  t.pop_state();
}

void test_08() {
  std::cout << "\ntest_08" << std::endl;
  using namespace aks;
  ag_d::tape_t t;
  const ag_d::var_t x = t.new_variable(2, true);
  const ag_d::var_t y = t.new_variable(3);

  ag_d::var_t f =
      (x ^ y) * exp(x) * exp(y) / (log(y) - (y ^ ag_d::value_t(-0.5))) +
      sin(cos(x) * y * ag_d::value_t(0.5));

  t.push_state();
  AKS_CHECK_VARIABLE(x, 2);
  AKS_CHECK_VARIABLE(y, 3);
  AKS_CHECK_VARIABLE(f, 2277.166893402295);
  vec_t<ag_d::value_t> expected = {
      5693.271663432962, 12529.22628253501, 24487.49989506043,
      43265.42685712199, 70607.95619169925, 108326.8348413039,
      157615.1751647243, 218966.3917658434, 304763.4639907167};

  for (int i = 0; i < 9; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(x);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 42734);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 11775);
  t.pop_state();
}

void test_09() {
  std::cout << "\ntest_09" << std::endl;
  using namespace aks;
  ag_d::tape_t t;
  const ag_d::var_t x = t.new_variable(2, true);
  const ag_d::var_t y = t.new_variable(3);

  ag_d::var_t f =
      sin(y * tanh(x / ag_d::value_t(4.0))) / tanh(y / ag_d::value_t(6.0));

  // AKS_PRINT(as_dot(t));

  t.push_state();
  AKS_CHECK_VARIABLE(x, 2);
  AKS_CHECK_VARIABLE(y, 3);
  AKS_CHECK_VARIABLE(f, 2.127248903688577);
  vec_t<ag_d::value_t> expected = {
      0.234088640404172, -0.794171421323257, 0.421052063524680,
      0.403203519062006, -0.966405900729128, 0.315910699048416,
      1.943245437957466, -3.596458737095583, -1.901001142886110};

  for (int i = 0; i < 9; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(x);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 64625);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 15510);
  t.pop_state();
}

void test_10() {
  std::cout << "\ntest_10" << std::endl;
  using namespace aks;
  ag_d::tape_t t;
  const ag_d::var_t x = t.new_variable(2, true);
  const ag_d::var_t y = t.new_variable(3);

  ag_d::var_t f =
      (x ^ y) * exp(x) * exp(y) / (log(y) - (ag_d::value_t(1.0) / sqrt(y))) +
      sin(cos(x) * y * ag_d::value_t(0.5));

  t.push_state();
  AKS_CHECK_VARIABLE(x, 2);
  AKS_CHECK_VARIABLE(y, 3);
  AKS_CHECK_VARIABLE(f, 2277.166893402295);
  vec_t<ag_d::value_t> expected = {
      5693.271663432962, 12529.22628253501, 24487.49989506043,
      43265.42685712199, 70607.95619169925, 108326.8348413039,
      157615.1751647243, 218966.3917658434, 304763.4639907167};

  for (int i = 0; i < 9; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(x);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 42735);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 11775);
  t.pop_state();
}

void test_11() {
  std::cout << "\ntest_11" << std::endl;
  using namespace aks;
  ag_d::tape_t t;
  const ag_d::var_t x = t.new_variable(0.5, true);

  ag_d::var_t f = relu(relu(relu(x)));

  t.push_state();
  AKS_CHECK_VARIABLE(x, 0.5);
  AKS_CHECK_VARIABLE(f, 0.5);
  vec_t<ag_d::value_t> expected = {1, 0, 0};

  for (int i = 0; i < 3; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(x);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 15);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 2);
  t.pop_state();
}

void test_12() {
  std::cout << "\ntest_12" << std::endl;
  using namespace aks;
  ag_d::tape_t t;
  const ag_d::var_t x = t.new_variable(-0.5, true);

  ag_d::var_t f = relu(relu(relu(x)));

  t.push_state();
  AKS_CHECK_VARIABLE(x, -0.5);
  AKS_CHECK_VARIABLE(f, 0);
  vec_t<ag_d::value_t> expected = {0, 0, 0};

  for (int i = 0; i < 3; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(x);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 15);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 2);
  t.pop_state();
}

void test_13() {
  std::cout << "\ntest_13" << std::endl;
  using namespace aks;
  ag_d::tape_t t;
  const ag_d::var_t x = t.new_variable(2, true);
  const ag_d::var_t y = t.new_variable(3);

  ag_d::var_t f =
      (x ^ y) * exp(x) * exp(y) / (log(y) - (ag_d::value_t(1.0) / sqrt(y))) +
      sin(cos(x) * y * ag_d::value_t(0.5));

  f = f * f;

  t.push_state();
  AKS_CHECK_VARIABLE(x, 2);
  AKS_CHECK_VARIABLE(y, 3);
  AKS_CHECK_VARIABLE(f, 5185489.060407462);
  vec_t<ag_d::value_t> expected = {
      25929059.49422991, 121888963.0483667, 539517981.8810239,
      2254246176.856630, 8920979033.396202, 33572462599.63692,
      120670307539.7625, 416043043171.6418, 1381489125066.02686};

  for (int i = 0; i < 9; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(x);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 193502);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 46561);
  t.pop_state();
}

void test_14() {
  std::cout << "\ntest_14" << std::endl;
  using namespace aks;
  ag_d::tape_t t;
  const ag_d::var_t x = t.new_variable(0.25, true);
  const ag_d::var_t y = t.new_variable(0.5);

  ag_d::var_t f =
      (x ^ y) / (log(y) - (ag_d::value_t(1.0) / sqrt(y))) + sin(cos(x));

  f = sqrt(f);

  t.push_state();
  AKS_CHECK_VARIABLE(x, 0.25);
  AKS_CHECK_VARIABLE(y, 0.5);
  AKS_CHECK_VARIABLE(f, 0.766163700778299);
  vec_t<ag_d::value_t> expected = {-0.401093405500755, 1.843951799064022e-02,
                                   -3.976997195889043, 27.94021527444540,
                                   -443.0134618423613, 7768.010873937546,
                                   -172271.1423942442, 4465339.597362671};

  for (int i = 0; i < 8; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(x);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 72391);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 16622);
  t.pop_state();
}

void test_15() {
  std::cout << "\ntest_15" << std::endl;
  using namespace aks;
  ag_d::tape_t t;
  const ag_d::var_t x = t.new_variable(2, true);
  const ag_d::var_t y = t.new_variable(3, true);
  const ag_d::var_t z = t.new_variable(5, true);

  auto F = [&]() {
    return ((x + z) ^ y) * exp(ag_d::value_t(0.25) * sqrt(z) * y * x) *
               exp(ag_d::value_t(0.25) * (y + x)) /
               (log(x + y) - (ag_d::value_t(1.0) / sqrt(x * y))) +
           sin(z * cos(x) * y * ag_d::value_t(0.5));
  };

  {
    t.push_state();
    ag_d::var_t f = F();

    AKS_CHECK_VARIABLE(x, 2);
    AKS_CHECK_VARIABLE(y, 3);
    AKS_CHECK_VARIABLE(z, 5);
    AKS_CHECK_VARIABLE(f, 28524.51801677098);
    vec_t<ag_d::value_t> expected = {60026.82047554181, 129114.1367025926,
                                     278095.5325188761, 602778.7382787745,
                                     1314895.486871684, 2690006.380087323,
                                     5081703.820602682, 25346219.35630465};

    for (int i = 0; i < 8; ++i) {
      t.zero_grad();
      backward(f);
      f = grad(x);
      AKS_CHECK_VARIABLE(f, expected[i]);
    }
    AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 259994);
    AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 58341);
    t.pop_state();
  }
  {
    t.push_state();
    ag_d::var_t f = F();

    AKS_CHECK_VARIABLE(x, 2);
    AKS_CHECK_VARIABLE(y, 3);
    AKS_CHECK_VARIABLE(z, 5);
    AKS_CHECK_VARIABLE(f, 28524.51801677098);
    vec_t<ag_d::value_t> expected = {88164.61349317656, 275673.9023954568,
                                     868833.1020929292, 2754733.9246692426};

    for (int i = 0; i < 4; ++i) {
      t.zero_grad();
      backward(f);
      f = grad(y);
      AKS_CHECK_VARIABLE(f, expected[i]);
    }
    AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 2084);
    AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 430);
    t.pop_state();
  }
  {
    t.push_state();
    ag_d::var_t f = F();

    AKS_CHECK_VARIABLE(x, 2);
    AKS_CHECK_VARIABLE(y, 3);
    AKS_CHECK_VARIABLE(z, 5);
    AKS_CHECK_VARIABLE(f, 28524.51801677098);
    vec_t<ag_d::value_t> expected = {21792.84737554386, 13945.72372321069,
                                     7309.719265308730, 3064.054319595382,
                                     1006.296282742556};

    for (int i = 0; i < 5; ++i) {
      t.zero_grad();
      backward(f);
      f = grad(z);
      AKS_CHECK_VARIABLE(f, expected[i]);
    }
    AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 2484);
    AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 482);
    t.pop_state();
  }
}

void test_16() {
  std::cout << "\ntest_16" << std::endl;

  using namespace aks;

  ag_d::tape_t t;

  const ag_d::var_t x = t.new_variable(2, true);
  const ag_d::var_t y = t.new_variable(3, true);
  const ag_d::var_t z = t.new_variable(4, true);

  auto F = [&]() {
    ag_d::var_t f = (z * (x + y));
    for (int i = 0; i < 3; ++i) {
      f *= f;
    }
    return f / ag_d::value_t(10000.0);
  };

  {

    t.push_state();
    ag_d::var_t f = F();

    AKS_CHECK_VARIABLE(x, 2);
    AKS_CHECK_VARIABLE(y, 3);
    AKS_CHECK_VARIABLE(z, 4);
    AKS_CHECK_VARIABLE(f, 2560000);

    vec_t<ag_d::value_t> expected = {
        4096000,     5734400,     6881280,     6881280, 5505024,
        3303014.400, 1321205.760, 264241.1520, 0,       0};

    for (int i = 0; i < 8; ++i) {
      t.zero_grad();
      backward(f);
      f = grad(x);
      AKS_CHECK_VARIABLE(f, expected[i]);
    }
    AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 11306);
    AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 2176);
    t.pop_state();
  }

  {

    t.push_state();
    ag_d::var_t f = F();

    AKS_CHECK_VARIABLE(x, 2);
    AKS_CHECK_VARIABLE(y, 3);
    AKS_CHECK_VARIABLE(z, 4);
    AKS_CHECK_VARIABLE(f, 2560000);

    vec_t<ag_d::value_t> expected = {
        4096000,     5734400,     6881280,     6881280, 5505024,
        3303014.400, 1321205.760, 264241.1520, 0,       0};

    for (int i = 0; i < 10; ++i) {
      t.zero_grad();
      backward(f);
      f = grad(y);
      AKS_CHECK_VARIABLE(f, expected[i]);
    }
    AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 17141);
    AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 2);
    t.pop_state();
  }

  {

    t.push_state();
    ag_d::var_t f = F();

    AKS_CHECK_VARIABLE(x, 2);
    AKS_CHECK_VARIABLE(y, 3);
    AKS_CHECK_VARIABLE(z, 4);
    AKS_CHECK_VARIABLE(f, 2560000);

    vec_t<ag_d::value_t> expected = {5120000,  8960000,  13440000, 16800000,
                                     16800000, 12600000, 6300000,  1575000,
                                     0,        0};

    for (int i = 0; i < 10; ++i) {
      t.zero_grad();
      backward(f);
      f = grad(z);
      AKS_CHECK_VARIABLE(f, expected[i]);
    }
    AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 17129);
    AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 2);
    t.pop_state();
  }
}

void test_17() {
  std::cout << "\ntest_17" << std::endl;

  using namespace aks;

  ag_d::tape_t t;

  const ag_d::var_t x = t.new_variable(2, true);
  const ag_d::var_t y = t.new_variable(3, true);
  const ag_d::var_t z = t.new_variable(4, true);

  auto F = [&]() {
    ag_d::var_t f = (z * (x + y));
    for (int i = 0; i < 2; ++i) {
      f *= f;
    }
    ag_d::var_t f2 = (z - x) * y;
    for (int i = 0; i < 1; ++i) {
      f2 *= f2;
    }
    return f / f2;

    // return ((z * (x + y)) ^ 4.0) / (((z - x) * y) ^ 2.0);
  };

  {
    t.push_state();
    ag_d::var_t f = F();

    AKS_CHECK_VARIABLE(x, 2);
    AKS_CHECK_VARIABLE(y, 3);
    AKS_CHECK_VARIABLE(z, 4);
    AKS_CHECK_VARIABLE(f, 4444.444444444444);

    vec_t<ag_d::value_t> expected = {
        8000,   15911.11111111111, 36586.66666666666,
        98784,  310986.6666666667, 1125040,
        4609920};

    for (int i = 0; i < 7; ++i) {
      t.zero_grad();
      backward(f);
      f = grad(x);
      // AKS_PRINT(f);
      AKS_CHECK_VARIABLE(f, expected[i]);
    }
    AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 20686);
    AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 4354);
    t.pop_state();
  }

  {

    t.push_state();
    ag_d::var_t f = F();

    AKS_CHECK_VARIABLE(x, 2);
    AKS_CHECK_VARIABLE(y, 3);
    AKS_CHECK_VARIABLE(z, 4);
    AKS_CHECK_VARIABLE(f, 4444.444444444444);

    vec_t<ag_d::value_t> expected = {592.5925925925926,  355.5555555555555,
                                     -252.8395061728395, 370.8312757201646,
                                     -674.2386831275720, 1460.850480109739,
                                     -3670.855052583448};

    for (int i = 0; i < 7; ++i) {
      t.zero_grad();
      backward(f);
      f = grad(y);
      AKS_CHECK_VARIABLE(f, expected[i]);
    }
    AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 20674);
    AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 4353);
    t.pop_state();
  }

  {

    t.push_state();
    ag_d::var_t f = F();

    AKS_CHECK_VARIABLE(x, 2);
    AKS_CHECK_VARIABLE(y, 3);
    AKS_CHECK_VARIABLE(z, 4);
    AKS_CHECK_VARIABLE(f, 4444.444444444444);

    vec_t<ag_d::value_t> expected = {
        0,      1111.111111111111,  -1666.666666666667,
        3750,   -10416.66666666667, 34375,
        -131250};

    for (int i = 0; i < 7; ++i) {
      t.zero_grad();
      backward(f);
      f = grad(z);
      AKS_CHECK_VARIABLE(f, expected[i]);
    }
    AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 20698);
    AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 4354);
    t.pop_state();
  }
}

void test_18() {
  std::cout << "\ntest_18" << std::endl;
  using namespace aks;
  ag_d::tape_t t;
  const ag_d::var_t x = t.new_variable(2, true);

  ag_d::var_t f = ((sin(x) ^ ag_d::value_t(2.0)) /
                   (log(x + ag_d::value_t(50.0)) ^ ag_d::value_t(2.0))) *
                      tanh((ag_d::value_t(1.0) - (exp(x) ^ (-x)))) +
                  cos(x);

  t.push_state();
  AKS_CHECK_VARIABLE(x, 2);
  AKS_CHECK_VARIABLE(f, -0.376226238713693);
  vec_t<ag_d::value_t> expected = {-0.944550600009225, 0.344604639299095,
                                   1.084903225704792,  -0.203449504037272,
                                   -1.677246351875284, 0.706056911049653,
                                   0.669408525907504,  -6.722648455610035};

  for (int i = 0; i < 8; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(x);
    // AKS_PRINT(f);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 308828);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 69864);
  t.pop_state();
}

void test_19() {
  std::cout << "\ntest_19" << std::endl;
  using namespace aks;

  auto NR = [](ag_d::value_t guess, auto f, ag_d::value_t tolerance = 1e-6) {
    ag_d::tape_t tape;

    auto derivative = [&tape](ag_d::var_t fx, ag_d::var_t x) {
      tape.push_state();
      tape.zero_grad();
      backward(fx);
      ag_d::var_t dfdx = grad(x, false);
      ag_d::value_t r_dfdx = dfdx.value();
      tape.pop_state();
      return r_dfdx;
    };

    ag_d::var_t x = tape.new_variable(guess, true);
    ag_d::var_t fx = f(x);

    if constexpr (aks::vcl_detail::is_vcl_vec<ag_d::value_t>::value) {
      using namespace aks::vcl_detail;
      while (vec_horizontal_or(abs(fx.value()) > tolerance)) {
        x = x - fx / derivative(fx, x);
        fx = f(x);
        // AKS_PRINT(x);
        // AKS_PRINT(fx);
      };
    } else {
      while (abs(fx.value()) > tolerance) {
        x = x - fx / derivative(fx, x);
        fx = f(x);
      };
    }

    return x.value();
  };

  AKS_CHECK_PRINT(
      "nr01", NR(3.0, [](auto x) { return x * x - ag_d::value_t(4.0); }), 2.0);
  AKS_CHECK_PRINT(
      "nr02", NR(3.0, [](auto x) { return x * x - ag_d::value_t(16.0); }), 4.0);
  AKS_CHECK_PRINT(
      "nr03", NR(5.0, [](auto x) { return x * x * x - ag_d::value_t(27.0); }),
      3.0);
  AKS_CHECK_PRINT(
      "nr04",
      NR(3.0,
         [](auto x) { return (x ^ ag_d::value_t(4.0)) - ag_d::value_t(16.0); }),
      2.0);
  AKS_CHECK_PRINT("nr05", NR(ag_d::value_t(0.2), [](auto x) { return sin(x); }),
                  ag_d::value_t(0.0));
  AKS_CHECK_PRINT("nr06", NR(ag_d::value_t(1.2), [](auto x) { return sin(x); }),
                  std::numbers::pi_v<double>);
}

void test_20() {
  std::cout << "\ntest_20" << std::endl;
  using namespace aks;

  ag_d::tape_t t;

  ag_d::var_t x = t.new_variable(2.0, true);
  ag_d::var_t y = t.new_variable(3.0, true);
  ag_d::var_t z = t.new_variable(5.0, true);

  AKS_CHECK_VARIABLE(x, 2.0);
  AKS_CHECK_VARIABLE(y, 3.0);
  AKS_CHECK_VARIABLE(z, 5.0);

  auto DIFF = [&](size_t I, size_t J, size_t K) {
    ag_d::tape_context_t context(t);
    ag_d::var_t f = (x * y * z) ^ ag_d::value_t(4.0);

    for (size_t i = 0; i < I; ++i) {
      t.zero_grad();
      backward(f);
      f = grad(x);
    }
    for (size_t j = 0; j < J; ++j) {
      t.zero_grad();
      backward(f);
      f = grad(y);
    }
    for (size_t k = 0; k < K; ++k) {
      t.zero_grad();
      backward(f);
      f = grad(z);
    }
    ag_d::value_t result = f.value();
    return result;
  };

  AKS_CHECK_VALUE(DIFF(0, 0, 0), 810000);
  AKS_CHECK_VALUE(DIFF(0, 0, 1), 648000);
  AKS_CHECK_VALUE(DIFF(0, 0, 2), 388800);
  AKS_CHECK_VALUE(DIFF(0, 0, 3), 155520);
  AKS_CHECK_VALUE(DIFF(0, 1, 0), 1080000);
  AKS_CHECK_VALUE(DIFF(0, 1, 1), 864000);
  AKS_CHECK_VALUE(DIFF(0, 1, 2), 518400);
  AKS_CHECK_VALUE(DIFF(0, 1, 3), 207360);
  AKS_CHECK_VALUE(DIFF(0, 2, 0), 1080000);
  AKS_CHECK_VALUE(DIFF(0, 2, 1), 864000);
  AKS_CHECK_VALUE(DIFF(0, 2, 2), 518400);
  AKS_CHECK_VALUE(DIFF(0, 2, 3), 207360);
  AKS_CHECK_VALUE(DIFF(0, 3, 0), 720000);
  AKS_CHECK_VALUE(DIFF(0, 3, 1), 576000);
  AKS_CHECK_VALUE(DIFF(0, 3, 2), 345600);
  AKS_CHECK_VALUE(DIFF(0, 3, 3), 138240);
  AKS_CHECK_VALUE(DIFF(1, 0, 0), 1620000);
  AKS_CHECK_VALUE(DIFF(1, 0, 1), 1296000);
  AKS_CHECK_VALUE(DIFF(1, 0, 2), 777600);
  AKS_CHECK_VALUE(DIFF(1, 0, 3), 311040);
  AKS_CHECK_VALUE(DIFF(1, 1, 0), 2160000);
  AKS_CHECK_VALUE(DIFF(1, 1, 1), 1728000);
  AKS_CHECK_VALUE(DIFF(1, 1, 2), 1036800);
  AKS_CHECK_VALUE(DIFF(1, 1, 3), 414720);
  AKS_CHECK_VALUE(DIFF(1, 2, 0), 2160000);
  AKS_CHECK_VALUE(DIFF(1, 2, 1), 1728000);
  AKS_CHECK_VALUE(DIFF(1, 2, 2), 1036800);
  AKS_CHECK_VALUE(DIFF(1, 2, 3), 414720);
  AKS_CHECK_VALUE(DIFF(1, 3, 0), 1440000);
  AKS_CHECK_VALUE(DIFF(1, 3, 1), 1152000);
  AKS_CHECK_VALUE(DIFF(1, 3, 2), 691200);
  AKS_CHECK_VALUE(DIFF(1, 3, 3), 276480);
  AKS_CHECK_VALUE(DIFF(2, 0, 0), 2430000);
  AKS_CHECK_VALUE(DIFF(2, 0, 1), 1944000);
  AKS_CHECK_VALUE(DIFF(2, 0, 2), 1166400);
  AKS_CHECK_VALUE(DIFF(2, 0, 3), 466560);
  AKS_CHECK_VALUE(DIFF(2, 1, 0), 3240000);
  AKS_CHECK_VALUE(DIFF(2, 1, 1), 2592000);
  AKS_CHECK_VALUE(DIFF(2, 1, 2), 1555200);
  AKS_CHECK_VALUE(DIFF(2, 1, 3), 622080);
  AKS_CHECK_VALUE(DIFF(2, 2, 0), 3240000);
  AKS_CHECK_VALUE(DIFF(2, 2, 1), 2592000);
  AKS_CHECK_VALUE(DIFF(2, 2, 2), 1555200);
  AKS_CHECK_VALUE(DIFF(2, 2, 3), 622080);
  AKS_CHECK_VALUE(DIFF(2, 3, 0), 2160000);
  AKS_CHECK_VALUE(DIFF(2, 3, 1), 1728000);
  AKS_CHECK_VALUE(DIFF(2, 3, 2), 1036800);
  AKS_CHECK_VALUE(DIFF(2, 3, 3), 414720);
  AKS_CHECK_VALUE(DIFF(3, 0, 0), 2430000);
  AKS_CHECK_VALUE(DIFF(3, 0, 1), 1944000);
  AKS_CHECK_VALUE(DIFF(3, 0, 2), 1166400);
  AKS_CHECK_VALUE(DIFF(3, 0, 3), 466560);
  AKS_CHECK_VALUE(DIFF(3, 1, 0), 3240000);
  AKS_CHECK_VALUE(DIFF(3, 1, 1), 2592000);
  AKS_CHECK_VALUE(DIFF(3, 1, 2), 1555200);
  AKS_CHECK_VALUE(DIFF(3, 1, 3), 622080);
  AKS_CHECK_VALUE(DIFF(3, 2, 0), 3240000);
  AKS_CHECK_VALUE(DIFF(3, 2, 1), 2592000);
  AKS_CHECK_VALUE(DIFF(3, 2, 2), 1555200);
  AKS_CHECK_VALUE(DIFF(3, 2, 3), 622080);
  AKS_CHECK_VALUE(DIFF(3, 3, 0), 2160000);
  AKS_CHECK_VALUE(DIFF(3, 3, 1), 1728000);
  AKS_CHECK_VALUE(DIFF(3, 3, 2), 1036800);
  AKS_CHECK_VALUE(DIFF(3, 3, 3), 414720);
}

void test_21() {
  std::cout << "\ntest_21" << std::endl;
  using namespace aks;

  ag_d::tape_t t;

  ag_d::var_t x = t.new_variable(2.0, true);
  ag_d::var_t y = t.new_variable(3.0, true);
  ag_d::var_t z = t.new_variable(5.0, true);

  AKS_CHECK_VARIABLE(x, 2.0);
  AKS_CHECK_VARIABLE(y, 3.0);
  AKS_CHECK_VARIABLE(z, 5.0);

  auto DIFF = [&](size_t I, size_t J, size_t K) {
    ag_d::tape_context_t ctx(t);
    ag_d::var_t f = ((z + (x * y)) ^ ag_d::value_t(4.0)) /
                    ((z - ((x * y) ^ ag_d::value_t(4.0))));
    for (size_t k = 0; k < K; ++k) {
      t.zero_grad();
      backward(f);
      f = grad(z);
    }
    for (size_t j = 0; j < J; ++j) {
      t.zero_grad();
      backward(f);
      f = grad(y);
    }
    for (size_t i = 0; i < I; ++i) {
      t.zero_grad();
      backward(f);
      f = grad(x);
    }

    ag_d::value_t result = f.value();
    return result;
  };

  AKS_CHECK_VALUE(DIFF(0, 0, 0), -11.34082106893881);
  AKS_CHECK_VALUE(DIFF(0, 0, 1), -4.132719458612656);
  AKS_CHECK_VALUE(DIFF(0, 0, 2), -1.131111881423102);
  AKS_CHECK_VALUE(DIFF(0, 1, 0), 6.931788386619875);
  AKS_CHECK_VALUE(DIFF(0, 1, 1), 3.287584053345693);
  AKS_CHECK_VALUE(DIFF(0, 1, 2), 1.110097985442148);
  AKS_CHECK_VALUE(DIFF(0, 2, 0), -7.875539780814895);
  AKS_CHECK_VALUE(DIFF(0, 2, 1), -4.093281799892223);
  AKS_CHECK_VALUE(DIFF(0, 2, 2), -1.538438319278641);
  AKS_CHECK_VALUE(DIFF(1, 0, 0), 10.39768257992981);
  AKS_CHECK_VALUE(DIFF(1, 0, 1), 4.931376080018539);
  AKS_CHECK_VALUE(DIFF(1, 0, 2), 1.665146978163221);
  AKS_CHECK_VALUE(DIFF(1, 1, 0), -8.347415477912405);
  AKS_CHECK_VALUE(DIFF(1, 1, 1), -4.496130673165488);
  AKS_CHECK_VALUE(DIFF(1, 1, 2), -1.752608486196887);
  AKS_CHECK_VALUE(DIFF(1, 2, 0), 10.53464584922905);
  AKS_CHECK_VALUE(DIFF(1, 2, 1), 6.082438639502456);
  AKS_CHECK_VALUE(DIFF(1, 2, 2), 2.571311648103607);
  AKS_CHECK_VALUE(DIFF(2, 0, 0), -17.71996450683351);
  AKS_CHECK_VALUE(DIFF(2, 0, 1), -9.209884049757502);
  AKS_CHECK_VALUE(DIFF(2, 0, 2), -3.461486218376942);
  AKS_CHECK_VALUE(DIFF(2, 1, 0), 15.80196877384358);
  AKS_CHECK_VALUE(DIFF(2, 1, 1), 9.123657959253684);
  AKS_CHECK_VALUE(DIFF(2, 1, 2), 3.856967472155410);
  AKS_CHECK_VALUE(DIFF(2, 2, 0), -21.40905253764527);
  AKS_CHECK_VALUE(DIFF(2, 2, 1), -13.13058746849400);
  AKS_CHECK_VALUE(DIFF(2, 2, 2), -5.962797901944007);
}

void test_22() {
  std::cout << "\ntest_22" << std::endl;
  using namespace aks;

  ag_d::tape_t t;

  ag_d::var_t x = t.new_variable(2.0, true);
  ag_d::var_t y = t.new_variable(3.0, true);
  ag_d::var_t z = t.new_variable(5.0, true);

  AKS_CHECK_VARIABLE(x, 2.0);
  AKS_CHECK_VARIABLE(y, 3.0);
  AKS_CHECK_VARIABLE(z, 5.0);

  auto DIFF = [&](size_t I, size_t J, size_t K) {
    ag_d::tape_context_t context(t);
    ag_d::var_t f =
        (z * (sin(x + y) + cos(x - y))) / (log(x) * tanh(y) * exp(z));
    for (size_t k = 0; k < K; ++k) {
      t.zero_grad();
      backward(f);
      f = grad(z);
    }
    for (size_t j = 0; j < J; ++j) {
      t.zero_grad();
      backward(f);
      f = grad(y);
    }
    for (size_t i = 0; i < I; ++i) {
      t.zero_grad();
      backward(f);
      f = grad(x);
    }

    ag_d::value_t result = f.value();
    return result;
  };

  AKS_CHECK_VALUE(DIFF(0, 0, 0), -2.044782741051969e-02);
  AKS_CHECK_VALUE(DIFF(0, 0, 1), 1.635826192841575e-02);
  AKS_CHECK_VALUE(DIFF(0, 0, 2), -1.226869644631181e-02);
  AKS_CHECK_VALUE(DIFF(0, 1, 0), -2.704374543052498e-02);
  AKS_CHECK_VALUE(DIFF(0, 1, 1), 2.163499634441998e-02);
  AKS_CHECK_VALUE(DIFF(0, 1, 2), -1.622624725831499e-02);
  AKS_CHECK_VALUE(DIFF(0, 2, 0), 2.058063059798511e-02);
  AKS_CHECK_VALUE(DIFF(0, 2, 1), -1.646450447838809e-02);
  AKS_CHECK_VALUE(DIFF(0, 2, 2), 1.234837835879106e-02);
  AKS_CHECK_VALUE(DIFF(1, 0, 0), 6.970775721602036e-02);
  AKS_CHECK_VALUE(DIFF(1, 0, 1), -5.576620577281628e-02);
  AKS_CHECK_VALUE(DIFF(1, 0, 2), 4.182465432961222e-02);
  AKS_CHECK_VALUE(DIFF(1, 1, 0), 9.219360095503580e-02);
  AKS_CHECK_VALUE(DIFF(1, 1, 1), -7.375488076402864e-02);
  AKS_CHECK_VALUE(DIFF(1, 1, 2), 5.531616057302147e-02);
  AKS_CHECK_VALUE(DIFF(1, 2, 0), -7.016049051445333e-02);
  AKS_CHECK_VALUE(DIFF(1, 2, 1), 5.612839241156267e-02);
  AKS_CHECK_VALUE(DIFF(1, 2, 2), -4.209629430867199e-02);
  AKS_CHECK_VALUE(DIFF(2, 0, 0), -8.749420303705228e-02);
  AKS_CHECK_VALUE(DIFF(2, 0, 1), 6.999536242964184e-02);
  AKS_CHECK_VALUE(DIFF(2, 0, 2), -5.249652182223137e-02);
  AKS_CHECK_VALUE(DIFF(2, 1, 0), -0.115717474823922);
  AKS_CHECK_VALUE(DIFF(2, 1, 1), 9.257397985913732e-02);
  AKS_CHECK_VALUE(DIFF(2, 1, 2), -6.943048489435298e-02);
  AKS_CHECK_VALUE(DIFF(2, 2, 0), 8.806245455907409e-02);
  AKS_CHECK_VALUE(DIFF(2, 2, 1), -7.044996364725926e-02);
  AKS_CHECK_VALUE(DIFF(2, 2, 2), 5.283747273544445e-02);
}

void test_23() {
  std::cout << "\ntest_23" << std::endl;
  using namespace aks;

  ag_d::tape_t t;

  ag_d::var_t x = t.new_variable(2.0, true);
  ag_d::var_t y = t.new_variable(3.0, true);
  ag_d::var_t z = t.new_variable(5.0, true);

  AKS_CHECK_VARIABLE(x, 2.0);
  AKS_CHECK_VARIABLE(y, 3.0);
  AKS_CHECK_VARIABLE(z, 5.0);

  auto DIFF = [&](size_t I, size_t J, size_t K) {
    ag_d::tape_context_t context(t);
    ag_d::var_t f =
        (((x * y) ^ ag_d::value_t(2.0)) + ((x ^ z) - (y ^ z))) / (sqrt(z - x));
    for (size_t k = 0; k < K; ++k) {
      t.zero_grad();
      backward(f);
      f = grad(z);
    }
    for (size_t j = 0; j < J; ++j) {
      t.zero_grad();
      backward(f);
      f = grad(y);
    }
    for (size_t i = 0; i < I; ++i) {
      t.zero_grad();
      backward(f);
      f = grad(x);
    }

    ag_d::value_t result = f.value();
    return result;
  };

  AKS_CHECK_VALUE(DIFF(0, 0, 0), -101.0362971081845);
  AKS_CHECK_VALUE(DIFF(0, 0, 1), -124.4856148327459);
  AKS_CHECK_VALUE(DIFF(0, 0, 2), -121.7651399067734);
  AKS_CHECK_VALUE(DIFF(0, 0, 3), -127.9635437257823);
  AKS_CHECK_VALUE(DIFF(0, 1, 0), -219.9704525612474);
  AKS_CHECK_VALUE(DIFF(0, 1, 1), -266.9886904528319);
  AKS_CHECK_VALUE(DIFF(0, 1, 2), -302.0851690012146);
  AKS_CHECK_VALUE(DIFF(0, 1, 3), -347.5287356773748);
  AKS_CHECK_VALUE(DIFF(0, 2, 0), -307.1503432088809);
  AKS_CHECK_VALUE(DIFF(0, 2, 1), -431.6178058676168);
  AKS_CHECK_VALUE(DIFF(0, 2, 2), -580.3877854573810);
  AKS_CHECK_VALUE(DIFF(0, 2, 3), -765.6492666607779);
  AKS_CHECK_VALUE(DIFF(0, 3, 0), -311.7691453623979);
  AKS_CHECK_VALUE(DIFF(0, 3, 1), -534.7710539628299);
  AKS_CHECK_VALUE(DIFF(0, 3, 2), -868.0046893093057);
  AKS_CHECK_VALUE(DIFF(0, 3, 3), -1346.101202148069);
  AKS_CHECK_VALUE(DIFF(1, 0, 0), 50.13324837463250);
  AKS_CHECK_VALUE(DIFF(1, 0, 1), 14.95612115044345);
  AKS_CHECK_VALUE(DIFF(1, 0, 2), 16.62282503236905);
  AKS_CHECK_VALUE(DIFF(1, 0, 3), 5.737428760076625);
  AKS_CHECK_VALUE(DIFF(1, 1, 0), -22.80533563299022);
  AKS_CHECK_VALUE(DIFF(1, 1, 1), -34.58693545438340);
  AKS_CHECK_VALUE(DIFF(1, 1, 2), -27.67447137673993);
  AKS_CHECK_VALUE(DIFF(1, 1, 3), -30.05453447975932);
  AKS_CHECK_VALUE(DIFF(1, 2, 0), -46.57292171462981);
  AKS_CHECK_VALUE(DIFF(1, 2, 1), -55.64219338080670);
  AKS_CHECK_VALUE(DIFF(1, 2, 2), -59.76480204884563);
  AKS_CHECK_VALUE(DIFF(1, 2, 3), -67.77925903137404);
  AKS_CHECK_VALUE(DIFF(1, 3, 0), -51.96152422706631);
  AKS_CHECK_VALUE(DIFF(1, 3, 1), -71.80800091811621);
  AKS_CHECK_VALUE(DIFF(1, 3, 2), -96.79544760614014);
  AKS_CHECK_VALUE(DIFF(1, 3, 3), -127.5547527518713);
  AKS_CHECK_VALUE(DIFF(2, 0, 0), 116.6728668987369);
  AKS_CHECK_VALUE(DIFF(2, 0, 1), 86.29947678117064);
  AKS_CHECK_VALUE(DIFF(2, 0, 2), 89.89945558016721);
  AKS_CHECK_VALUE(DIFF(2, 0, 3), 68.29891300984247);
  AKS_CHECK_VALUE(DIFF(2, 1, 0), -6.783865662978102);
  AKS_CHECK_VALUE(DIFF(2, 1, 1), -13.49257845502667);
  AKS_CHECK_VALUE(DIFF(2, 1, 2), -5.227083564478602);
  AKS_CHECK_VALUE(DIFF(2, 1, 3), -9.543583555761231);
  AKS_CHECK_VALUE(DIFF(2, 2, 0), -21.74686013947590);
  AKS_CHECK_VALUE(DIFF(2, 2, 1), -20.05894307129838);
  AKS_CHECK_VALUE(DIFF(2, 2, 2), -16.63807237004382);
  AKS_CHECK_VALUE(DIFF(2, 2, 3), -17.16602377242994);
  AKS_CHECK_VALUE(DIFF(2, 3, 0), -25.98076211353316);
  AKS_CHECK_VALUE(DIFF(2, 3, 1), -27.24374642121372);
  AKS_CHECK_VALUE(DIFF(2, 3, 2), -30.23522618892759);
  AKS_CHECK_VALUE(DIFF(2, 3, 3), -33.54215018700807);
  AKS_CHECK_VALUE(DIFF(3, 0, 0), 199.6749868484843);
  AKS_CHECK_VALUE(DIFF(3, 0, 1), 203.3328691179788);
  AKS_CHECK_VALUE(DIFF(3, 0, 2), 247.3726161180334);
  AKS_CHECK_VALUE(DIFF(3, 0, 3), 237.9811544790943);
  AKS_CHECK_VALUE(DIFF(3, 1, 0), -8.347522642033338);
  AKS_CHECK_VALUE(DIFF(3, 1, 1), -7.883957562654816);
  AKS_CHECK_VALUE(DIFF(3, 1, 2), 0.547243573532937);
  AKS_CHECK_VALUE(DIFF(3, 1, 3), -8.152750541321677);
  AKS_CHECK_VALUE(DIFF(3, 2, 0), -19.02048386830267);
  AKS_CHECK_VALUE(DIFF(3, 2, 1), -10.18317451358455);
  AKS_CHECK_VALUE(DIFF(3, 2, 2), -7.193885687481728);
  AKS_CHECK_VALUE(DIFF(3, 2, 3), -6.995307679983575);
  AKS_CHECK_VALUE(DIFF(3, 3, 0), -21.65063509461097);
  AKS_CHECK_VALUE(DIFF(3, 3, 1), -15.48624365280778);
  AKS_CHECK_VALUE(DIFF(3, 3, 2), -14.87185938890113);
  AKS_CHECK_VALUE(DIFF(3, 3, 3), -13.07993243360558);
}

void test_24_01() {
  std::cout << "\ntest_24_01" << std::endl;
  using namespace aks;

  ag_d::tape_t t;

  ag_d::var_t x = t.new_variable(2.0, true);
  ag_d::var_t y = t.new_variable(3.0, true);
  ag_d::var_t z = t.new_variable(5.0, true);

  AKS_CHECK_VARIABLE(x, 2.0);
  AKS_CHECK_VARIABLE(y, 3.0);
  AKS_CHECK_VARIABLE(z, 5.0);

  auto DIFF = [&](size_t I, size_t J, size_t K) {
    t.push_state();
    ag_d::var_t f = tanh(x * tanh(sqrt(z - x)));
    ;
    for (size_t k = 0; k < K; ++k) {
      t.zero_grad();
      backward(f);
      f = grad(z);
    }
    for (size_t j = 0; j < J; ++j) {
      t.zero_grad();
      backward(f);
      f = grad(y);
    }
    for (size_t i = 0; i < I; ++i) {
      t.zero_grad();
      backward(f);
      f = grad(x);
    }

    ag_d::value_t result = f.value();
    t.pop_state();
    return result;
  };

  AKS_CHECK_VALUE(DIFF(0, 0, 0), 0.954367034850222);
  AKS_CHECK_VALUE(DIFF(0, 1, 0), 0.0);
  AKS_CHECK_VALUE(DIFF(0, 0, 1), 6.061400669882047e-03);
  AKS_CHECK_VALUE(DIFF(0, 0, 2), -5.083687521844842e-03);
  AKS_CHECK_VALUE(DIFF(0, 0, 3), 5.581847524380858e-03);
  AKS_CHECK_VALUE(DIFF(1, 0, 0), 7.770852538096278e-02);
  AKS_CHECK_VALUE(DIFF(1, 0, 1), -2.752914043067545e-03);
  AKS_CHECK_VALUE(DIFF(1, 0, 2), 5.285110160304726e-04);
  AKS_CHECK_VALUE(DIFF(1, 0, 3), 1.640651071941994e-03);
  AKS_CHECK_VALUE(DIFF(2, 0, 0), -0.139599050715691);
  AKS_CHECK_VALUE(DIFF(2, 0, 1), 1.023603858623439e-03);
  AKS_CHECK_VALUE(DIFF(2, 0, 2), -7.076421643329067e-04);
  AKS_CHECK_VALUE(DIFF(2, 0, 3), 2.038101125111268e-03);
  AKS_CHECK_VALUE(DIFF(3, 0, 0), 0.245847487120853);
  AKS_CHECK_VALUE(DIFF(3, 0, 1), 4.655063058225343e-03);
  AKS_CHECK_VALUE(DIFF(3, 0, 2), -4.438653791965176e-03);
  AKS_CHECK_VALUE(DIFF(3, 0, 3), 6.993946507113833e-03);
}

void test_24_02() {
  std::cout << "\ntest_24_02" << std::endl;
  using namespace aks;

  ag_d::tape_t t;

  ag_d::var_t x = t.new_variable(2.0, true);
  ag_d::var_t y = t.new_variable(3.0, true);
  ag_d::var_t z = t.new_variable(5.0, true);

  AKS_CHECK_VARIABLE(x, 2.0);
  AKS_CHECK_VARIABLE(y, 3.0);
  AKS_CHECK_VARIABLE(z, 5.0);

  auto DIFF = [&](size_t I, size_t J, size_t K) {
    t.push_state();
    ag_d::var_t f = sqrt(x * exp(y - log(z)));
    for (size_t k = 0; k < K; ++k) {
      t.zero_grad();
      backward(f);
      f = grad(z);
    }
    for (size_t j = 0; j < J; ++j) {
      t.zero_grad();
      backward(f);
      f = grad(y);
    }
    for (size_t i = 0; i < I; ++i) {
      t.zero_grad();
      backward(f);
      f = grad(x);
    }

    ag_d::value_t result = f.value();
    t.pop_state();
    return result;
  };

  AKS_CHECK_VALUE(DIFF(0, 0, 0), 2.834469045390171);
  AKS_CHECK_VALUE(DIFF(0, 0, 1), -0.283446904539017);
  AKS_CHECK_VALUE(DIFF(0, 0, 2), 8.503407136170513e-02);
  AKS_CHECK_VALUE(DIFF(0, 0, 3), -4.251703568085256e-02);
  AKS_CHECK_VALUE(DIFF(0, 1, 0), 1.417234522695086);
  AKS_CHECK_VALUE(DIFF(0, 1, 1), -0.141723452269509);
  AKS_CHECK_VALUE(DIFF(0, 1, 2), 4.251703568085256e-02);
  AKS_CHECK_VALUE(DIFF(0, 1, 3), -2.125851784042628e-02);
  AKS_CHECK_VALUE(DIFF(0, 2, 0), 0.708617261347543);
  AKS_CHECK_VALUE(DIFF(0, 2, 1), -7.086172613475428e-02);
  AKS_CHECK_VALUE(DIFF(0, 2, 2), 2.125851784042628e-02);
  AKS_CHECK_VALUE(DIFF(0, 2, 3), -1.062925892021314e-02);
  AKS_CHECK_VALUE(DIFF(0, 3, 0), 0.354308630673771);
  AKS_CHECK_VALUE(DIFF(0, 3, 1), -3.543086306737714e-02);
  AKS_CHECK_VALUE(DIFF(0, 3, 2), 1.062925892021314e-02);
  AKS_CHECK_VALUE(DIFF(0, 3, 3), -5.314629460106570e-03);
  AKS_CHECK_VALUE(DIFF(1, 0, 0), 0.708617261347543);
  AKS_CHECK_VALUE(DIFF(1, 0, 1), -7.086172613475428e-02);
  AKS_CHECK_VALUE(DIFF(1, 0, 2), 2.125851784042628e-02);
  AKS_CHECK_VALUE(DIFF(1, 0, 3), -1.062925892021314e-02);
  AKS_CHECK_VALUE(DIFF(1, 1, 0), 0.354308630673771);
  AKS_CHECK_VALUE(DIFF(1, 1, 1), -3.543086306737714e-02);
  AKS_CHECK_VALUE(DIFF(1, 1, 2), 1.062925892021314e-02);
  AKS_CHECK_VALUE(DIFF(1, 1, 3), -5.314629460106570e-03);
  AKS_CHECK_VALUE(DIFF(1, 2, 0), 0.177154315336886);
  AKS_CHECK_VALUE(DIFF(1, 2, 1), -1.771543153368857e-02);
  AKS_CHECK_VALUE(DIFF(1, 2, 2), 5.314629460106570e-03);
  AKS_CHECK_VALUE(DIFF(1, 2, 3), -2.657314730053285e-03);
  AKS_CHECK_VALUE(DIFF(1, 3, 0), 8.857715766844285e-02);
  AKS_CHECK_VALUE(DIFF(1, 3, 1), -8.857715766844285e-03);
  AKS_CHECK_VALUE(DIFF(1, 3, 2), 2.657314730053285e-03);
  AKS_CHECK_VALUE(DIFF(1, 3, 3), -1.328657365026643e-03);
  AKS_CHECK_VALUE(DIFF(2, 0, 0), -0.177154315336886);
  AKS_CHECK_VALUE(DIFF(2, 0, 1), 1.771543153368857e-02);
  AKS_CHECK_VALUE(DIFF(2, 0, 2), -5.314629460106570e-03);
  AKS_CHECK_VALUE(DIFF(2, 0, 3), 2.657314730053285e-03);
  AKS_CHECK_VALUE(DIFF(2, 1, 0), -8.857715766844285e-02);
  AKS_CHECK_VALUE(DIFF(2, 1, 1), 8.857715766844285e-03);
  AKS_CHECK_VALUE(DIFF(2, 1, 2), -2.657314730053285e-03);
  AKS_CHECK_VALUE(DIFF(2, 1, 3), 1.328657365026643e-03);
  AKS_CHECK_VALUE(DIFF(2, 2, 0), -4.428857883422142e-02);
  AKS_CHECK_VALUE(DIFF(2, 2, 1), 4.428857883422142e-03);
  AKS_CHECK_VALUE(DIFF(2, 2, 2), -1.328657365026643e-03);
  AKS_CHECK_VALUE(DIFF(2, 2, 3), 6.643286825133213e-04);
  AKS_CHECK_VALUE(DIFF(2, 3, 0), -2.214428941711071e-02);
  AKS_CHECK_VALUE(DIFF(2, 3, 1), 2.214428941711071e-03);
  AKS_CHECK_VALUE(DIFF(2, 3, 2), -6.643286825133213e-04);
  AKS_CHECK_VALUE(DIFF(2, 3, 3), 3.321643412566607e-04);
  AKS_CHECK_VALUE(DIFF(3, 0, 0), 0.132865736502664);
  AKS_CHECK_VALUE(DIFF(3, 0, 1), -1.328657365026643e-02);
  AKS_CHECK_VALUE(DIFF(3, 0, 2), 3.985972095079927e-03);
  AKS_CHECK_VALUE(DIFF(3, 0, 3), -1.992986047539964e-03);
  AKS_CHECK_VALUE(DIFF(3, 1, 0), 6.643286825133213e-02);
  AKS_CHECK_VALUE(DIFF(3, 1, 1), -6.643286825133213e-03);
  AKS_CHECK_VALUE(DIFF(3, 1, 2), 1.992986047539964e-03);
  AKS_CHECK_VALUE(DIFF(3, 1, 3), -9.964930237699818e-04);
  AKS_CHECK_VALUE(DIFF(3, 2, 0), 3.321643412566606e-02);
  AKS_CHECK_VALUE(DIFF(3, 2, 1), -3.321643412566607e-03);
  AKS_CHECK_VALUE(DIFF(3, 2, 2), 9.964930237699818e-04);
  AKS_CHECK_VALUE(DIFF(3, 2, 3), -4.982465118849909e-04);
  AKS_CHECK_VALUE(DIFF(3, 3, 0), 1.660821706283303e-02);
  AKS_CHECK_VALUE(DIFF(3, 3, 1), -1.660821706283303e-03);
  AKS_CHECK_VALUE(DIFF(3, 3, 2), 4.982465118849909e-04);
  AKS_CHECK_VALUE(DIFF(3, 3, 3), -2.491232559424955e-04);
}

void test_25() {
  std::cout << "\ntest_25" << std::endl;
  using namespace aks;

  ag_d::tape_t t;

  auto to_variable = [&](ag_d::value_t v) { return t.new_variable(v, true); };

  vec_t<ag_d::var_t> xs =
      zipped_op(to_variable, vec_t<ag_d::value_t>{2.0, 3.0, 5.0});
  vec_t<ag_d::var_t> ys =
      zipped_op(to_variable, vec_t<ag_d::value_t>{7.0, 11.0, 13.0});

  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 6);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 0);

  ag_d::var_t d = dot(xs, ys);

  AKS_CHECK_VARIABLE(d, 112.0);

  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 7);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 0);

  t.zero_grad();
  backward(d);
  AKS_CHECK_VARIABLE(grad(xs[0]), 7.0);
  AKS_CHECK_VARIABLE(grad(ys[1]), 3.0);

  ag_d::var_t s = asum(ys);
  AKS_CHECK_VARIABLE(s, 31.0);

  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 15);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 7);

  t.zero_grad();
  backward(s);
  AKS_CHECK_VARIABLE(grad(xs[0]), 0.0);
  AKS_CHECK_VARIABLE(grad(ys[1]), 1.0);

  ag_d::var_t g = gsum(xs);
  AKS_CHECK_VARIABLE(g, 30.0);

  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 24);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 5);

  // backward(g);
  // AKS_CHECK_VARIABLE(grad(xs[0]), 15.0);

  AKS_CHECK_VARIABLE(g, 30.0);

  if constexpr (!aks::vcl_detail::is_vcl_vec<ag_d::value_t>::value) {

    ag_d::var_t mx = max(xs);
    AKS_CHECK_VARIABLE(mx, 5.0);

    AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 24);
    AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 5);

    // backward(mx);
    // AKS_CHECK_VARIABLE(grad(xs[0]), 0.0);

    ag_d::var_t mn = min(xs);
    AKS_CHECK_VARIABLE(mn, 2.0);

    AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 24);
    AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 5);

    // backward(mn);
    // AKS_CHECK_VARIABLE(grad(xs[0]), 1.0);

    ag_d::var_t mm = mean(ys);
    AKS_CHECK_VARIABLE(mm, 10.3333333);

    AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 27);
    AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 5);

    // backward(mm);
    // AKS_CHECK_VARIABLE(grad(ys[0]), 1.0);

    ag_d::var_t gm = gmean(xs);
    AKS_CHECK_VARIABLE(gm, 10.0);

    AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 30);
    AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 5);

    // backward(gm);
    // AKS_CHECK_VARIABLE(grad(xs[0]), 1.0);
  }
}

void test_26() {
  std::cout << "\ntest_26" << std::endl;

#ifndef AKS_NO_VCL

  using ag = autograd_traits<vcl::Vec2d>;

  using namespace aks;
  using namespace vcl;
  using namespace std;
  ag::tape_t t;
  const ag::var_t x = t.new_variable(
      ag::value_t(std::numbers::pi_v<double_t>) / ag::value_t(2.0, 1.0), true);

  ag::var_t f = (sin(x) ^ ag::value_t(2.0)) + (cos(x) ^ ag::value_t(2.0));

  t.push_state();
  AKS_CHECK_PRINT(x.value()[0], x.value()[0],
                  std::numbers::pi_v<double_t> / double_t(2.0));
  AKS_CHECK_PRINT(x.value()[1], x.value()[1],
                  std::numbers::pi_v<double_t> / double_t(1.0));
  AKS_CHECK_PRINT(f.value()[0], f.value()[0], double_t(1.0));
  AKS_CHECK_PRINT(f.value()[1], f.value()[1], double_t(1.0));

  for (int i = 0; i < 1; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(x);
    AKS_CHECK_PRINT(f.value()[0], f.value()[0], 0.0);
    AKS_CHECK_PRINT(f.value()[1], f.value()[1], 0.0);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 30);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 6);
  t.pop_state();

#endif
}

void test_27() {
  std::cout << "\ntest_27" << std::endl;
#ifndef AKS_NO_VCL

  auto test_with = [](auto X, auto Y) {
    using namespace aks;
    using r_t = decltype(X);
    using f_t = decltype(Y);
    using ag = autograd_traits<r_t>;
    using tape_type = typename ag::tape_t;
    using var_type = typename ag::var_t;
    using value_type = typename ag::value_t;

    using namespace aks;
    using namespace vcl;
    using namespace std;
    tape_type t;
    const var_type x = t.new_variable(
        value_type(std::numbers::pi_v<f_t>) / value_type(2.0), true);

    var_type f = sin(x);

    t.push_state();
    vec_t<value_type> expected = {0, -1, 0, 1, 0, -1, 0, 1, 0};

    for (int i = 0; i < 9; ++i) {
      t.zero_grad();
      backward(f);
      f = grad(x);
      AKS_CHECK_PRINT(f.value()[0], f.value()[0], expected[i]);
      AKS_CHECK_PRINT(f.value()[1], f.value()[1], expected[i]);
    }
    AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 49);
    AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 4);

    // AKS_PRINT(as_dot(t));

    t.pop_state();
  };

  test_with(vcl::Vec2d(), double_t());
  test_with(vcl::Vec4d(), double_t());
  test_with(vcl::Vec8d(), double_t());
  test_with(vcl::Vec4f(), float_t());
  test_with(vcl::Vec8f(), float_t());
  test_with(vcl::Vec16f(), float_t());

#endif
}

void test_28() {
  std::cout << "\ntest_28" << std::endl;
#ifndef AKS_NO_VCL
  using namespace vcl;
  using namespace aks;
  using namespace aks::vcl_detail;
  using Vec = vcl::Vec16f;
  using Vecb = vcl::Vec16fb;

  using ag = autograd_traits<Vec>;

  auto NR = [](Vec guess, auto f, auto expected_tape_sizes,
               Vec tolerance = 1e-6f) {
    ag::tape_t tape;

    auto derivative = [&](ag::var_t fx, ag::var_t x) {
      tape.push_state();
      tape.zero_grad();
      backward(fx);
      ag::var_t dfdx = grad(x, false);
      ag::value_t r_dfdx = dfdx.value();

      AKS_CHECK_PRINT(tape.nodes_.size(), tape.nodes_.size(),
                      expected_tape_sizes[2]);

      tape.pop_state();
      return r_dfdx;
    };

    ag::var_t x = tape.new_variable(guess, true);

    size_t iter = 0;
    size_t max_iter = 100;

    Vecb not_solved(true);

    while (vec_horizontal_or(not_solved) && iter++ < max_iter) {
      tape.push_state();
      ag::var_t fx = f(x);

      not_solved = abs(fx.value()) > tolerance;

      ag::var_t y = x - fx / derivative(fx, x);

      // only update the ones that need solving
      Vec new_x = vcl::select(not_solved, y.value(), x.value());

      x.update_in_place(new_x);

      AKS_CHECK_PRINT(tape.nodes_.size(), tape.nodes_.size(),
                      expected_tape_sizes[1]);
      tape.pop_state();
      AKS_CHECK_PRINT(tape.nodes_.size(), tape.nodes_.size(),
                      expected_tape_sizes[0]);
    };

    return x.value();
  };

  auto test_NR = [&](auto guess, auto f, auto expected,
                     auto expected_tape_sizes) {
    const Vec result = NR(guess, f, expected_tape_sizes);

    for (int i = 0; i < result.size(); ++i) {
      AKS_CHECK_PRINT("nr", result[i], expected[i]);
    }
  };

  const Vec N(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
              11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f);

  test_NR(
      Vec(5.0f), [&](ag::var_t x) { return (x * x) - N; },
      Vec(1.f, 1.41421353816986f, 1.73205077648163f, 2, 2.2360680103302f,
          2.44948983192444f, 2.64575123786926f, 2.82842707633972f, 3.f,
          3.16227769851685f, 3.31662487983704f, 3.46410155296326f,
          3.60555124282837f, 3.74165749549866f, 3.87298321723938f, 4.f),
      vec_t<size_t>{1, 7, 10});

  test_NR(
      Vec(3.0), [&](ag::var_t x) { return (x ^ Vec(3.0f)) - N; },
      Vec(1.f, 1.25992107391357f, 1.44224953651428f, 1.58740103244781f,
          1.70997595787048f, 1.81712055206299f, 1.91293120384216f, 2.f,
          2.0800838470459f, 2.15443468093872f, 2.22398018836975f,
          2.28942847251892f, 2.35133457183838f, 2.41014218330383f,
          2.46621203422546f, 2.51984214782715f),
      vec_t<size_t>{1, 8, 13});

#endif
}

void test_29() {
  std::cout << "\ntest_29" << std::endl;

  using namespace aks;

  ag_d::tape_t t;

  ag_d::var_t x = t.new_variable(2, true);
  ag_d::var_t y = t.new_variable(3, true);
  ag_d::var_t z = x + y;

  ag_d::var_t f = (x ^ y) * z;

  backward(f);

  ag_d::var_t dfdx = grad(x);
  ag_d::var_t dfdy = grad(y);
  ag_d::var_t dfdz = grad(z, false);

  AKS_CHECK_VARIABLE(z, 5);
  AKS_CHECK_VARIABLE(f, 40);
  AKS_CHECK_VARIABLE(dfdx, 68);
  AKS_CHECK_VARIABLE(dfdy, 35.7258872223978);
  AKS_CHECK_VARIABLE(dfdz, 8);
  AKS_CHECK_VALUE(t.nodes_.size(), 18);

  x.update_in_place(3);

  forward(&t);

  AKS_CHECK_VARIABLE(z, 6);
  AKS_CHECK_VARIABLE(f, 162);
  AKS_CHECK_VARIABLE(dfdx, 189);
  AKS_CHECK_VARIABLE(dfdy, 204.975190764234);
  AKS_CHECK_VARIABLE(dfdz, 27);
  AKS_CHECK_VALUE(t.nodes_.size(), 18);

  y.update_in_place(4);

  forward_from(&t, &y);

  AKS_CHECK_VARIABLE(z, 7);
  AKS_CHECK_VARIABLE(f, 567);
  AKS_CHECK_VARIABLE(dfdx, 837);
  AKS_CHECK_VARIABLE(dfdy, 703.9131676748182834);
  AKS_CHECK_VARIABLE(dfdz, 81);
  AKS_CHECK_VALUE(t.nodes_.size(), 18);

  y.update_in_place(5);
  // z.update_in_place(5); //this will fail, can't update a non-leaf node
  // "safely"

  forward_to(&t, &z);

  AKS_CHECK_VARIABLE(z, 8);
  AKS_CHECK_VARIABLE(f, 567);
  AKS_CHECK_VARIABLE(dfdx, 837);
  AKS_CHECK_VARIABLE(dfdy, 703.9131676748182834);
  AKS_CHECK_VARIABLE(dfdz, 81);
  AKS_CHECK_VALUE(t.nodes_.size(), 18);

  forward_from(&t, &z, false /*start from a non-leaf*/);

  AKS_CHECK_VARIABLE(z, 8);
  AKS_CHECK_VARIABLE(f, 1944);
  AKS_CHECK_VARIABLE(dfdx, 3483);
  AKS_CHECK_VARIABLE(dfdy, 2378.70228917080522);
  AKS_CHECK_VARIABLE(dfdz, 243);
  AKS_CHECK_VALUE(t.nodes_.size(), 18);

  forward(&t);

  AKS_CHECK_VARIABLE(z, 8);
  AKS_CHECK_VARIABLE(f, 1944);
  AKS_CHECK_VARIABLE(dfdx, 3483);
  AKS_CHECK_VARIABLE(dfdy, 2378.70228917080522);
  AKS_CHECK_VARIABLE(dfdz, 243);
  AKS_CHECK_VALUE(t.nodes_.size(), 18);
}

void test_30() {
  std::cout << "\ntest_30" << std::endl;

  using namespace aks;

  ag_d::tape_t t;

  ag_d::var_t x = t.new_variable(-0.4, true);
  ag_d::var_t y = t.new_variable(0.3, true);

  ag_d::var_t z = x + y;

  ag_d::var_t f = ag_d::value_t(1.0) * relu(z);

  backward(f);

  ag_d::var_t dfdx = grad(x);
  ag_d::var_t dfdy = grad(y);

  // AKS_PRINT(as_dot(t));

  AKS_CHECK_VARIABLE(x, -0.4);
  AKS_CHECK_VARIABLE(y, 0.3);
  AKS_CHECK_VARIABLE(z, -0.1);
  AKS_CHECK_VARIABLE(f, 0.0);
  AKS_CHECK_VARIABLE(dfdx, 0.0);
  AKS_CHECK_VARIABLE(dfdy, 0.0);
  AKS_CHECK_VALUE(t.nodes_.size(), 14);

  y.update_in_place(0.5);

  // in rerun of forward, all the conditions taken before will be the same, so
  // relu 0 * x will still be 0 * x
  forward(&t);

  // AKS_PRINT(as_dot(t));

  AKS_CHECK_VARIABLE(x, -0.4);
  AKS_CHECK_VARIABLE(y, 0.5);
  AKS_CHECK_VARIABLE(z, 0.1);
  AKS_CHECK_VARIABLE(f, 0.1);
  AKS_CHECK_VARIABLE(dfdx, 1.0);
  AKS_CHECK_VARIABLE(dfdy, 1.0);
  AKS_CHECK_VALUE(t.nodes_.size(), 14);
}

void test_31() {
  std::cout << "\ntest_31" << std::endl;
#ifndef AKS_NO_VCL
  using Vec = vcl::Vec4f;
  using ag = autograd_traits<Vec>;

  using namespace aks;

  ag::tape_t t;

  ag::var_t x = t.new_variable(Vec(-0.1f, -1.2f, -1.3f, -0.4f), true);
  ag::var_t y = t.new_variable(Vec(1.2f, 1.0f, 1.5f, -0.4f), true);

  ag::var_t f = relu(relu(relu(relu(x + y))));

  backward(f);

  ag::var_t dfdx = grad(x);
  ag::var_t dfdy = grad(y);

  AKS_CHECK_VALUE(f.value()[0], 1.1f);
  AKS_CHECK_VALUE(f.value()[1], 0.0f);
  AKS_CHECK_VALUE(f.value()[2], 0.2f);
  AKS_CHECK_VALUE(f.value()[3], 0.0f);
  AKS_CHECK_VALUE(dfdx.value()[0], 1.f);
  AKS_CHECK_VALUE(dfdx.value()[1], 0.f);
  AKS_CHECK_VALUE(dfdx.value()[2], 1.f);
  AKS_CHECK_VALUE(dfdx.value()[3], 0.f);
  AKS_CHECK_VALUE(dfdy.value()[0], 1.f);
  AKS_CHECK_VALUE(dfdy.value()[1], 0.f);
  AKS_CHECK_VALUE(dfdy.value()[2], 1.f);
  AKS_CHECK_VALUE(dfdy.value()[3], 0.f);

  // AKS_PRINT(as_dot(t));

  x.update_in_place(Vec(-0.4f, -0.1f, -1.2f, -1.3f));
  y.update_in_place(Vec(-0.4f, 1.2f, 1.0f, 1.5f));

  forward(&t);

  AKS_CHECK_VALUE(f.value()[0], 0.0f);
  AKS_CHECK_VALUE(f.value()[1], 1.1f);
  AKS_CHECK_VALUE(f.value()[2], 0.0f);
  AKS_CHECK_VALUE(f.value()[3], 0.2f);

  AKS_CHECK_VALUE(dfdx.value()[0], 0.f);
  AKS_CHECK_VALUE(dfdx.value()[1], 1.f);
  AKS_CHECK_VALUE(dfdx.value()[2], 0.f);
  AKS_CHECK_VALUE(dfdx.value()[3], 1.f);

  AKS_CHECK_VALUE(dfdy.value()[0], 0.f);
  AKS_CHECK_VALUE(dfdy.value()[1], 1.f);
  AKS_CHECK_VALUE(dfdy.value()[2], 0.f);
  AKS_CHECK_VALUE(dfdy.value()[3], 1.f);

  // AKS_PRINT(as_dot(t));
#endif
}

void test_32() {
  std::cout << "\ntest_32" << std::endl;

  using Vec = float_t;
  using ag = autograd_traits<Vec>;

  using namespace aks;

  ag::tape_t t;

  ag::var_t x = t.new_variable(Vec(-0.1f), true);
  ag::var_t y = t.new_variable(Vec(1.2f), true);

  ag::var_t f = relu(relu(relu(relu(x + y))));

  backward(f);

  ag::var_t dfdx = grad(x);
  ag::var_t dfdy = grad(y);

  AKS_CHECK_VALUE(f.value(), 1.1f);

  AKS_CHECK_VALUE(dfdx.value(), 1.f);
  AKS_CHECK_VALUE(dfdy.value(), 1.f);

  // AKS_PRINT(as_dot(t));

  x.update_in_place(Vec(-0.4f));
  y.update_in_place(Vec(-0.4f));

  forward(&t);

  AKS_CHECK_VALUE(f.value(), 0.0f);
  AKS_CHECK_VALUE(dfdx.value(), 0.f);
  AKS_CHECK_VALUE(dfdy.value(), 0.f);

  // AKS_PRINT(as_dot(t));
}

void test_33() {
  std::cout << "\ntest_33" << std::endl;

  using namespace aks;

  using ag = autograd_traits<double_t>;

  auto NR = [](ag::value_t guess, auto f, ag::value_t tolerance = 1e-6) {
    ag::tape_t tape;

    ag::var_t x = tape.new_variable(guess, true);
    ag::var_t fx;
    ag::var_t y;

    fx = f(x);
    bool not_solved = (abs(fx.value()) > tolerance);

    if (not_solved) {
      backward(fx);
      y = x - (fx / grad(x));
      x.update_in_place(y.value());
    }

    const size_t size_to_check = tape.nodes_.size();

    while (not_solved) {
      forward_to(&tape, &fx);
      not_solved = (abs(fx.value()) > tolerance);
      if (not_solved) {
        // we are doing a partial completion, so safe=false
        forward_from(&tape, &fx, false);
        x.update_in_place(y.value());
      }
      AKS_CHECK_VALUE(tape.nodes_.size(), size_to_check);
    };

    return x.value();
  };

  auto nr01 = NR(3.0, [](auto x) { return x * x - ag::value_t(4.0); });
  AKS_CHECK_VALUE(nr01, 2.0);

  auto nr02 = NR(3.0, [](auto x) { return x * x - ag::value_t(16.0); });
  AKS_CHECK_VALUE(nr02, 4.0);

  auto nr03 = NR(5.0, [](auto x) { return x * x * x - ag::value_t(27.0); });
  AKS_CHECK_VALUE(nr03, 3.0);

  auto nr04 = NR(
      3.0, [](auto x) { return (x ^ ag::value_t(4.0)) - ag::value_t(16.0); });
  AKS_CHECK_VALUE(nr04, 2.0);

  auto nr05 = NR(ag::value_t(0.2), [](auto x) { return sin(x); });
  AKS_CHECK_VALUE(nr05, ag::value_t(0.0));

  auto nr06 = NR(ag::value_t(1.2), [](auto x) { return sin(x); });
  AKS_CHECK_VALUE(nr06, std::numbers::pi_v<double>);
}

void test_34() {
  std::cout << "\ntest_34" << std::endl;

  using namespace aks;

  using ag = autograd_traits<double_t>;

  ag::tape_t tape;

  auto to_new_variable = [&](ag::value_t x) {
    return tape.new_variable(x, true);
  };
  auto in_place_update = [&](ag::var_t v, ag::value_t x) {
    return v.update_in_place(x);
  };

  vec_t<ag::var_t> xs =
      zipped_op(to_new_variable, vec_t<ag::value_t>{2.0, 3.0, 5.0});
  vec_t<ag::var_t> ys =
      zipped_op(to_new_variable, vec_t<ag::value_t>{1.0, 2.0, 3.0});

  var_t f = relu(dot(xs, ys));

  backward(f);

  AKS_CHECK_VARIABLE(f, 23.0);
  AKS_CHECK_VARIABLE(grad(xs[0]), 1.0);
  AKS_CHECK_VARIABLE(grad(xs[1]), 2.0);
  AKS_CHECK_VARIABLE(grad(xs[2]), 3.0);
  AKS_CHECK_VARIABLE(grad(ys[0]), 2.0);
  AKS_CHECK_VARIABLE(grad(ys[1]), 3.0);
  AKS_CHECK_VARIABLE(grad(ys[2]), 5.0);

  // AKS_PRINT(as_dot(tape));

  zipped_op_in_place(in_place_update, xs, vec_t<ag::value_t>{-1.0, 2.0, -3.0});
  zipped_op_in_place(in_place_update, ys, vec_t<ag::value_t>{1.0, -2.0, 3.0});

  forward(&tape);

  AKS_CHECK_VARIABLE(f, 0.0);
  AKS_CHECK_VARIABLE(grad(xs[0]), 0.0);
  AKS_CHECK_VARIABLE(grad(xs[1]), 0.0);
  AKS_CHECK_VARIABLE(grad(xs[2]), 0.0);
  AKS_CHECK_VARIABLE(grad(ys[0]), 0.0);
  AKS_CHECK_VARIABLE(grad(ys[1]), 0.0);
  AKS_CHECK_VARIABLE(grad(ys[2]), 0.0);

  // AKS_PRINT(as_dot(tape));

  zipped_op_in_place(in_place_update, xs, vec_t<ag::value_t>{-1.0, -2.0, -3.0});
  zipped_op_in_place(in_place_update, ys, vec_t<ag::value_t>{-2.0, -3.0, -5.0});

  forward(&tape);

  AKS_CHECK_VARIABLE(f, 23.0);
  AKS_CHECK_VARIABLE(grad(xs[0]), -2.0);
  AKS_CHECK_VARIABLE(grad(xs[1]), -3.0);
  AKS_CHECK_VARIABLE(grad(xs[2]), -5.0);
  AKS_CHECK_VARIABLE(grad(ys[0]), -1.0);
  AKS_CHECK_VARIABLE(grad(ys[1]), -2.0);
  AKS_CHECK_VARIABLE(grad(ys[2]), -3.0);

  // AKS_PRINT(as_dot(tape));
}

void test_35() {
  std::cout << "\ntest_35" << std::endl;

  using namespace aks;

  aks::vec_t<double_t> Xs = {2.0, 5.0, 7.0};
  aks::vec_t<aks::vec_t<double_t>> expected = {{0.0496, 0.06, 0.12, 1.0, 0.16},
                                               {0.0604, 0.06, 0.3, 1.0, 0.34},
                                               {0.0676, 0.06, 0.42, 1.0, 0.46}};

  mlp nn(1, {1, 1});

  ag_mlp::vec_var_t out(1), buffer(1);
  ag_mlp::vec_var_t x(1);

  ag_mlp::vec_var_t params = nn.parameters();

  params[0].update_in_place(0.04);
  params[1].update_in_place(0.06);
  params[2].update_in_place(0.04);
  params[3].update_in_place(0.06);

  size_t expected_idx = 0;
  for (double_t X : Xs) {
    if (x[0].is_alive()) {
      x[0].update_in_place(X);
    } else {
      x[0] = nn.tape.new_variable(X);
    }

    if (!out[0].is_alive()) {
      nn.forward(x, out, buffer);
      aks::backward(out[0]);
    } else {
      aks::forward(&nn.tape);
    }

    size_t expected_idx2 = 0;
    AKS_CHECK_VARIABLE(out[0], (expected[expected_idx][expected_idx2]));
    ++expected_idx2;
    for (auto &p : nn.parameters()) {
      AKS_CHECK_VARIABLE(grad(p), (expected[expected_idx][expected_idx2]));
      ++expected_idx2;
    }

    ++expected_idx;
  }
}

void test_36() {
  std::cout << "\ntest_36" << std::endl;

  using namespace aks;

  const double_t learning_rate = 0.1;
  size_t count = 0;
  size_t const max_iterations = 150;
  vec_t<double_t> losses;

  const vec_t<vec_t<double_t>> Xs = {
      {2.0, 3.0, -1.0}, {3.0, -1.0, 0.5}, {0.5, 1.0, 1.0}, {1.0, 1.0, -1.0}};
  const vec_t<double_t> Ys = {1.0, -1.0, -1.0, 1.0};

  mlp nn(3, {2, 1}, 55220);

  auto loss_func = [&](ag_mlp::vec_var_t const &y,
                       ag_mlp::vec_var_t const &pred) {
    ag_mlp::var_t two = nn.tape.new_variable(2.0);
    ag_mlp::vec_var_t each(y.size());

    for (size_t i = 0; i < y.size(); ++i) {
      each[i] = (pred[i] - y[i]) ^ two;
    }
    return mean(each);
  };

  ag_mlp::vec_var_t ys =
      zipped_op([&](double_t x) { return nn.tape.new_variable(x); }, Ys);
  ag_mlp::vec_var_t buffer, pred;
  ag_mlp::vec_var_t x;
  ag_mlp::var_t loss;
  ag_mlp::vec_var_t preds;

  ag_mlp::vec_var_t params = nn.parameters();

  while (count++ < max_iterations) {
    if (!loss.is_alive()) {
      for (size_t i = 0; i < Xs.size(); ++i) {
        x.resize(Xs[i].size());
        for (size_t j = 0; j < Xs[i].size(); ++j) {
          x[j] = nn.tape.new_variable(Xs[i][j]);
        }
        nn.forward(x, pred, buffer);
        preds.push_back(pred[0]);
      }
      loss = loss_func(ys, preds);
      nn.tape.zero_grad();
      aks::backward(loss);

    } else {
      for (size_t i = 0; i < Xs.size(); ++i) {
        for (size_t j = 0; j < Xs[i].size(); ++j) {
          x[j].update_in_place(Xs[i][j]);
        }
        aks::forward(&nn.tape);
      }
    }

    losses.push_back(loss.value());

    for (size_t i = 0; i < params.size(); ++i) {
      ag_mlp::var_t g = grad(params[i]);
      double_t update = params[i].value() - (learning_rate * g.value());
      params[i].update_in_place(update);
    }
  }

  AKS_CHECK_VALUE(losses.front(), 1.18089);
  AKS_CHECK_VALUE(losses.back(), 0.0);
}

void test_37() {
  std::cout << "\ntest_37" << std::endl;

  using namespace aks;

  ag_d::tape_t tape;

  {
    ag_d::tape_context_t context(tape);

    ag_d::var_t x = tape.new_variable(2.0, true);
    ag_d::var_t y = tape.new_variable(3.0, true);

    ag_d::var_t z = ((x * y) + (x * y));

    backward(z);

    ag_d::var_t dfdx = grad(x);
    ag_d::var_t dfdy = grad(y);

    AKS_CHECK_VARIABLE(dfdx, 6.0);
    AKS_CHECK_VARIABLE(dfdy, 4.0);
  }
  {
    ag_d::tape_context_t context(tape);

    ag_d::var_t x = tape.new_variable(2.0, true);
    ag_d::var_t y = tape.new_variable(3.0, true);

    ag_d::var_t z = ((x * y) + (x * y));

    auto const &[dfdx, dfdy] = gradient(z, x, y);

    AKS_CHECK_VARIABLE(dfdx, 6.0);
    AKS_CHECK_VARIABLE(dfdy, 4.0);
  }
}

void test_38() {
  std::cout << "\ntest_38" << std::endl;

  using namespace aks;

  ag_d::tape_t tape;

  ag_d::var_t x = tape.new_variable(ag_d::value_t(1.0), true);

  ag_d::var_t z =
      (x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x);

  AKS_CHECK_VARIABLE(z, ag_d::value_t(1.0));
  AKS_CHECK_VALUE(tape.nodes_.size(), 16);

  vec_t<ag_d::value_t> expected = {16.0,
                                   240.0,
                                   3360.0,
                                   43680.0,
                                   524160.0,
                                   5765760.0,
                                   57657600.0,
                                   518918400.0,
                                   4151347200.0,
                                   29059430400.0,
                                   174356582400.0,
                                   871782912000.0,
                                   3487131648000.0,
                                   10461394944000.0,
                                   20922789888000.0};

  vec_t<size_t> expected_size = {62,     266,     820,     2218,    5572,
                                 13310,  30584,   68018,   146860,  308134,
                                 627616, 1236890, 2342804, 4210574, 6995848};

  vec_t<size_t> expected_grads_size = {16,     60,     170,    474,    1238,
                                       3044,   7122,   15984,  34574,  72204,
                                       145418, 281096, 516102, 880644, 1327106};

  for (int i = 0; i < 12; ++i) {
    z = gradient(z, x);

    AKS_CHECK_VARIABLE(z, expected[i]);
    AKS_CHECK_VALUE(tape.nodes_.size(), expected_size[i]);
    AKS_CHECK_VALUE(tape.grads_.size(), expected_grads_size[i]);
  }
}

void test_39() {
  std::cout << "\ntest_39" << std::endl;

  using namespace aks;
  const double_t PI = std::numbers::pi_v<double_t>;

  vec_t<double_t> Xs_temp = {-1.0,
                             -0.7894736842105263,
                             -0.5789473684210527,
                             -0.368421052631579,
                             -0.1578947368421053,
                             0.05263157894736836,
                             0.26315789473684204,
                             0.4736842105263157,
                             0.6842105263157894,
                             0.894736842105263};
  for (double_t &x : Xs_temp) {
    x *= PI;
  }

  vec_t<vec_t<double_t>> Xs;
  for (double_t const &x : Xs_temp) {
    Xs.push_back({x});
  }

  vec_t<vec_t<double_t>> Ys;
  for (double_t &x : Xs_temp) {
    Ys.push_back({std::sin(x), std::cos(x)});
  }

  double_t learning_rate = 0.1;
  size_t count = 0;
  size_t const max_iterations = 10000;
  vec_t<double_t> losses;

  mlp nn(1, {8, 8, 2}, 55320);

  auto loss_func = [&](ag_mlp::vec_vec_var_t const &y,
                       ag_mlp::vec_vec_var_t const &pred) {
    ag_mlp::var_t two = nn.tape.new_variable(2.0);
    ag_mlp::vec_var_t each;

    for (size_t i = 0; i < y.size(); ++i) {
      for (size_t j = 0; j < y[i].size(); ++j) {
        each.push_back((pred[i][j] - y[i][j]) ^ two);
      }
    }
    return mean(each);
  };

  ag_mlp::vec_vec_var_t ys = zipped_op(
      [&](vec_t<double_t> xs) {
        return zipped_op([&](double_t x) { return nn.tape.new_variable(x); },
                         xs);
      },
      Ys);
  ag_mlp::vec_var_t buffer, pred;
  ag_mlp::vec_var_t x;
  ag_mlp::var_t loss;
  ag_mlp::vec_vec_var_t preds;

  optimizer opt(&nn.tape, nn.parameters(), learning_rate);

  AKS_CHECK_VALUE(opt.params_.size(), 106);

  while (count++ < max_iterations) {
    if (!loss.is_alive()) {
      for (size_t i = 0; i < Xs.size(); ++i) {
        x.resize(Xs[i].size());
        for (size_t j = 0; j < Xs[i].size(); ++j) {
          x[j] = nn.tape.new_variable(Xs[i][j]);
        }
        nn.forward(x, pred, buffer);
        preds.push_back(pred);
      }
      loss = loss_func(ys, preds);
      opt.zero_grad();
      aks::backward(loss);
    } else {
      for (size_t i = 0; i < Xs.size(); ++i) {
        for (size_t j = 0; j < Xs[i].size(); ++j) {
          x[j].update_in_place(Xs[i][j]);
        }
        aks::forward(&nn.tape);
      }
    }

    double_t const current_loss = loss.value();

    losses.push_back(current_loss);

    if (count % 1000 == 0) {
      if (learning_rate > 0.0001) {
        learning_rate *= 0.99;
      }
    }

    opt.step(learning_rate);
  }

  // AKS_CHECK_VALUE(losses.front(), 0.475618744490630185);
  // AKS_CHECK_VALUE(losses.back(), 0.0);

  AKS_CHECK_VALUE(losses.front(), 0.499967997191235);
  AKS_CHECK_VALUE(losses.back(), 0.0);

  if (false) {
    mlp_inf nn_inf(nn);
    std::vector<std::vector<double>> outs;
    for (double i = -PI * 3; i <= PI * 3; i += 0.1) {
      std::vector<double> op = {i};
      auto pred = nn_inf.forward(op);

      op.insert(op.end(), pred.begin(), pred.end());
      outs.push_back(op);
    }
    std::cout << "x"
              << ","
              << "pred_sin"
              << ","
              << "pred_cos"
              << ","
              << "sin"
              << ","
              << "cos" << std::endl;
    for (const auto &o : outs) {
      std::cout << o[0] << "," << o[1] << "," << o[2] << "," << std::sin(o[0])
                << "," << std::cos(o[0]) << std::endl;
    }
  }
}

void test_40() {
  std::cout << "\ntest_40" << std::endl;
  using namespace aks;
  using ag = autograd_traits<double_t>;

  {
    ag::vec_value_t xs = linspace(-1.0, 1.0, 5);
    ag::vec_value_t expected = {-1., -0.5, 0., 0.5, 1.};

    AKS_CHECK_VALUES(xs, expected);
  }

  {
    ag::vec_value_t xs = linspace(-1.0, 1.0, 5, false);
    ag::vec_value_t expected = {-1., -0.6, -0.2, 0.2, 0.6};

    AKS_CHECK_VALUES(xs, expected);
  }

  {
    ag::vec_value_t xs = linspace(-1.0, 1.0, 4);
    ag::vec_value_t expected = {-1., -0.33333333, 0.33333333, 1.};

    AKS_CHECK_VALUES(xs, expected);
  }

  {
    ag::vec_value_t xs = linspace(-1.0, 1.0, 4, false);
    ag::vec_value_t expected = {-1., -0.5, 0., 0.5};

    AKS_CHECK_VALUES(xs, expected);
  }
}

void test_41() {
  std::cout << "\ntest_41" << std::endl;

  using namespace aks;
  using ag = autograd_traits<double_t>;

  ag::tape_t tape;

  auto to_tape = [&](ag::vec_value_t const &vs, bool requires_grad = false) {
    return put_on_tape(&tape, vs, requires_grad);
  };

  auto var = [&](ag::value_t v, bool requires_grad = false) {
    return tape.new_variable(v, requires_grad);
  };

  {
    ag::tape_context_t context(tape);

    ag::vec_value_t data = linspace(-1.0, 1.0, 5);
    ag::vec_var_t xs = to_tape(data);

    AKS_CHECK_VARIABLES(xs, data);
  }

  {
    ag::tape_context_t context(tape);

    ag::vec_value_t data_x = linspace(-1.0, 1.0, 5);
    ag::vec_value_t data_y = linspace(1.0, -1.0, 5);
    ag::vec_var_t xs = to_tape(data_x);
    ag::vec_var_t ys = to_tape(data_y);

    AKS_CHECK_VARIABLES(xs, data_x);
    AKS_CHECK_VARIABLES(ys, data_y);

    ag::var_t f = dot(xs, ys);

    AKS_CHECK_VARIABLE(f, -2.5);
  }

  {
    ag::tape_context_t context(tape);

    ag::vec_value_t data_x = {1., 2., 3.};
    ag::vec_value_t data_y = {4., 5., 6.};
    ag::vec_var_t xs = to_tape(data_x, true);
    ag::vec_var_t ys = to_tape(data_y);

    AKS_CHECK_VARIABLES(xs, data_x);
    AKS_CHECK_VARIABLES(ys, data_y);

    ag::var_t f = dot(xs, ys);

    AKS_CHECK_VARIABLE(f, 32.);

    ag::vec_var_t gs = gradient(f, xs);
    AKS_CHECK_VARIABLES(gs, data_y);
  }

  {
    ag::tape_context_t context(tape);

    ag::vec_value_t data_x = {1., 2., 3.};
    ag::vec_value_t data_y = {4., 5., 6.};
    ag::vec_var_t xs = to_tape(data_x);
    ag::vec_var_t ys = to_tape(data_y, true);

    AKS_CHECK_VARIABLES(xs, data_x);
    AKS_CHECK_VARIABLES(ys, data_y);

    ag::var_t f = dot(xs, ys);

    AKS_CHECK_VARIABLE(f, 32.);

    ag::vec_var_t gs = gradient(f, ys);
    AKS_CHECK_VARIABLES(gs, data_x);
  }

  {
    ag::tape_context_t context(tape);

    ag::vec_value_t data_x = {1., 2., 3.};
    ag::vec_value_t data_y = {4., 5., 6.};
    ag::vec_var_t xs = to_tape(data_x, true);
    ag::vec_var_t ys = to_tape(data_y, true);

    AKS_CHECK_VARIABLES(xs, data_x);
    AKS_CHECK_VARIABLES(ys, data_y);

    ag::var_t f = dot(xs, ys);

    AKS_CHECK_VARIABLE(f, 32.);

    ag::vec_vec_var_t gs = gradient(f, {xs, ys});
    AKS_CHECK_VARIABLES(gs[0], data_y);
    AKS_CHECK_VARIABLES(gs[1], data_x);
  }

  {
    ag::tape_context_t context(tape);

    ag::var_t x = var(5.0, true);
    ag::var_t y = var(3.0);
    ag::var_t z = var(2.0);

    AKS_CHECK_VARIABLE(x, 5.0);
    AKS_CHECK_VARIABLE(y, 3.0);
    AKS_CHECK_VARIABLE(z, 2.0);

    ag::var_t f = (x + y) ^ z;

    AKS_CHECK_VARIABLE(f, 64.);

    ag::var_t gx = gradient(f, x);
    AKS_CHECK_VARIABLE(gx, 16.0);
  }

  {
    ag::tape_context_t context(tape);

    ag::var_t x = var(5.0, true);
    ag::var_t y = var(3.0, true);
    ag::var_t z = var(2.0, true);

    AKS_CHECK_VARIABLE(x, 5.0);
    AKS_CHECK_VARIABLE(y, 3.0);
    AKS_CHECK_VARIABLE(z, 2.0);

    ag::var_t f = (x + y) ^ z;

    AKS_CHECK_VARIABLE(f, 64.);

    auto [gx, gy, gz] = gradient(f, x, y, z);
    AKS_CHECK_VARIABLE(gx, 16.0);
    AKS_CHECK_VARIABLE(gy, 16.0);
    AKS_CHECK_VARIABLE(gz, 133.084258667509488);
  }

  {
    ag::tape_context_t context(tape);

    ag::var_t x = var(5.0, true);
    ag::var_t y = var(3.0, true);
    ag::var_t z = var(2.0, true);

    AKS_CHECK_VARIABLE(x, 5.0);
    AKS_CHECK_VARIABLE(y, 3.0);
    AKS_CHECK_VARIABLE(z, 2.0);

    ag::var_t fx = (x + y) ^ z;
    ag::var_t fy = (x + y) ^ z;
    ag::var_t fz = (x + y) ^ z;

    AKS_CHECK_VARIABLE(fx, 64.);
    AKS_CHECK_VARIABLE(fy, 64.);
    AKS_CHECK_VARIABLE(fz, 64.);

    ag::vec_var_t gs =
        gradient(ag::vec_var_t{fx, fy, fz}, ag::vec_var_t{x, y, z});
    AKS_CHECK_VARIABLE(gs[0], 16.0);
    AKS_CHECK_VARIABLE(gs[1], 16.0);
    AKS_CHECK_VARIABLE(gs[2], 133.084258667509488);
  }

  {
    ag::tape_context_t context(tape);

    ag::var_t x = var(5.0, true);
    ag::var_t y = var(3.0, true);
    ag::var_t z = var(2.0, true);

    AKS_CHECK_VARIABLE(x, 5.0);
    AKS_CHECK_VARIABLE(y, 3.0);
    AKS_CHECK_VARIABLE(z, 2.0);

    ag::var_t f = (x + y) ^ z;

    AKS_CHECK_VARIABLE(f, 64.);

    auto gs = gradient_map(f, {x, y, z});
    AKS_CHECK_VARIABLE(gs[x], 16.0);
    AKS_CHECK_VARIABLE(gs[y], 16.0);
    AKS_CHECK_VARIABLE(gs[z], 133.084258667509488);
  }

  {
    ag::tape_context_t context(tape);

    ag::var_t x = var(5.0, true);
    ag::var_t y = var(3.0, true);
    ag::var_t z = var(2.0, true);

    AKS_CHECK_VARIABLE(x, 5.0);
    AKS_CHECK_VARIABLE(y, 3.0);
    AKS_CHECK_VARIABLE(z, 2.0);

    ag::var_t fx = (x + y) ^ z;
    ag::var_t fy = (x + y) ^ z;
    ag::var_t fz = (x + y) ^ z;

    AKS_CHECK_VARIABLE(fx, 64.);
    AKS_CHECK_VARIABLE(fy, 64.);
    AKS_CHECK_VARIABLE(fz, 64.);

    auto gs = gradient_map(ag::vec_var_t{fx, fy, fz}, ag::vec_var_t{x, y, z});
    AKS_CHECK_VARIABLE(gs[x], 16.0);
    AKS_CHECK_VARIABLE(gs[y], 16.0);
    AKS_CHECK_VARIABLE(gs[z], 133.084258667509488);
  }

  {
    ag::tape_context_t context(tape);

    ag::var_t x = var(5.0, true);
    ag::var_t y = var(3.0, true);
    ag::var_t z = var(2.0, true);

    AKS_CHECK_VARIABLE(x, 5.0);
    AKS_CHECK_VARIABLE(y, 3.0);
    AKS_CHECK_VARIABLE(z, 2.0);

    ag::var_t f = (x + y) ^ z;

    AKS_CHECK_VARIABLE(f, 64.);

    AKS_CHECK_EQUAL(tape.nodes_.size(), 5);

    build_gradients(f);

    AKS_CHECK_EQUAL(tape.nodes_.size(), 18);

    {
      auto gs = get_grads_map(ag::vec_var_t{x, y, z});
      AKS_CHECK_VARIABLE(gs[x], 16.0);
      AKS_CHECK_VARIABLE(gs[y], 16.0);
      AKS_CHECK_VARIABLE(gs[z], 133.084258667509488);

      AKS_CHECK_EQUAL(tape.nodes_.size(), 18);
    }
    {
      auto gs = get_grads(ag::vec_var_t{x, y, z});
      AKS_CHECK_VARIABLE(gs[0], 16.0);
      AKS_CHECK_VARIABLE(gs[1], 16.0);
      AKS_CHECK_VARIABLE(gs[2], 133.084258667509488);

      AKS_CHECK_EQUAL(tape.nodes_.size(), 18);
    }
    {
      auto [gx, gy, gz] = get_grads(x, y, z);
      AKS_CHECK_VARIABLE(gx, 16.0);
      AKS_CHECK_VARIABLE(gy, 16.0);
      AKS_CHECK_VARIABLE(gz, 133.084258667509488);

      AKS_CHECK_EQUAL(tape.nodes_.size(), 18);
    }
  }
}

int main_tests() {
  test_41();
  test_40();
#ifdef NDEBUG
  test_39();
#endif
  test_38();
  test_37();
  test_36();
  test_35();
  test_34();
  test_33();
  test_32();
  test_31();
  test_30();
  test_29();
  test_28();
  test_27();
  test_26();
  test_25();
  test_24_02();
  test_24_01();
  test_23();
  test_22();
  test_21();
  test_20();
  test_19();
  test_18();
  test_17();
  test_16();
  test_15();
  test_14();
  test_13();
  test_12();
  test_11();
  test_10();
  test_09();
  test_08();
  test_07();
  test_06_02();
  test_06_01();
  test_05();
  test_04();
  test_03();
  test_02_f();
  test_02();
  test_01();

  std::cout << "\ntotal  : " << TOTAL_TEST_RUN
            << "\npassed : " << TOTAL_TEST_PASS
            << "\nfailed : " << TOTAL_TEST_FAIL << std::endl;
  return 0;
}

} // namespace

int main() {
  main_tests();
  return 0;
}
