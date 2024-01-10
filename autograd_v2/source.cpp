
#include "autograd_v2.hpp"

namespace {
constexpr bool QUIET_PASS = true;
constexpr bool QUIET_FAIL = true;
constexpr bool ASSERT_FAIL = false;
static size_t TOTAL_TEST_RUN = 0;
static size_t TOTAL_TEST_PASS = 0;
static size_t TOTAL_TEST_FAIL = 0;

#define AKS_PRINT(EXPR)                                                        \
  std::cout << std::setprecision(15) << #EXPR << " = " << EXPR << std::endl

#define AKS_PRINT_AS(NAME, EXPR)                                               \
  std::cout << std::setprecision(15) << NAME << " " << EXPR << "       \t("    \
            << #EXPR << ")" << std::endl

#define AKS_CHECK_PRINT(EXPR, EXPR_VAL, EXPECTED)                              \
  do {                                                                         \
    ++TOTAL_TEST_RUN;                                                          \
    const re_t diff__ = std::abs(re_t(EXPR_VAL) - re_t(EXPECTED));             \
    if (std::isnan(re_t(EXPR_VAL)) || diff__ > 1e-4) {                         \
      ++TOTAL_TEST_FAIL;                                                       \
      if (!QUIET_FAIL) {                                                       \
        std::cout << std::setprecision(18) << "\nCHECK FAILED: " << #EXPR      \
                  << " = " << re_t(EXPR_VAL) << " != " << EXPECTED << " ("     \
                  << diff__ << ")"                                             \
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
                  << " = " << re_t(EXPR_VAL) << std::endl;                     \
      } else {                                                                 \
        std::cout << ".";                                                      \
      }                                                                        \
    }                                                                          \
  } while (false)

#define AKS_CHECK_VARIABLE(EXPR, EXPECTED)                                     \
  AKS_CHECK_PRINT(EXPR, EXPR.value(), EXPECTED)

#define AKS_CHECK_VALUE(EXPR, EXPECTED) AKS_CHECK_PRINT(EXPR, (EXPR), EXPECTED)

} // namespace

namespace {

void test_01() {
  std::cout << "\ntest_01" << std::endl;
  using namespace aks;

  tape_t<re_t> t;

  const var_t x = t.new_variable(3.0);
  var_t y = t.new_variable(5.0);

  var_t f = (x * x * x * x * x * x * x * x);

  t.push_state();
  AKS_CHECK_VARIABLE(x, 3);
  AKS_CHECK_VARIABLE(y, 5);
  AKS_CHECK_VARIABLE(f, 6561);
  vec_t<re_t> expected = {17496,  40824,  81648, 136080, 181440,
                          181440, 120960, 40320, 0};

  for (int i = 0; i < 9; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(x);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 1175495);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 346633);
  t.pop_state();

  f = (y * y) * (x * x) * (y * y * y);

  expected = {28125, 22500, 13500, 5400, 1080, 0, 0, 0, 0};
  AKS_CHECK_VARIABLE(f, 28125);
  t.push_state();
  for (int i = 0; i < 9; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(y);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 1332723);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 390635);
  t.pop_state();

  f = (x * x) / (y * y);

  expected = {-0.144, 0.0864, -0.06912, 0.06912, -0.0829439999999999};
  AKS_CHECK_VARIABLE(f, 0.36);
  t.push_state();
  for (int i = 0; i < 5; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(y);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 14507);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 3783);
  t.pop_state();
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 18);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 0);
}

void test_02() {
  std::cout << "\ntest_02" << std::endl;
  using namespace aks;
  tape_t<re_t> t;
  const var_t x = t.new_variable(3.0);
  const var_t y = t.new_variable(8.0);

  var_t f = pow(x, y);

  t.push_state();
  AKS_CHECK_VARIABLE(x, 3);
  AKS_CHECK_VARIABLE(y, 8);
  AKS_CHECK_VARIABLE(f, 6561);
  vec_t<re_t> expected = {17496,  40824,  81648, 136080, 181440,
                          181440, 120960, 40320, 0};

  for (int i = 0; i < 9; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(x);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 456774);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 126296);
  t.pop_state();
}

void test_03() {
  std::cout << "\ntest_03" << std::endl;
  using namespace aks;
  tape_t<re_t> t;
  const var_t x = t.new_variable(3.0);

  var_t f = pow(x, 8.0);

  t.push_state();
  AKS_CHECK_VARIABLE(x, 3);
  AKS_CHECK_VARIABLE(f, 6561);
  vec_t<re_t> expected = {17496,  40824,  81648, 136080, 181440,
                          181440, 120960, 40320, 0};

  for (int i = 0; i < 9; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(x);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 456774);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 126296);
  t.pop_state();
}

void test_04() {
  std::cout << "\ntest_04" << std::endl;
  using namespace aks;
  tape_t<re_t> t;
  const var_t x = t.new_variable(3.0);

  var_t f = exp(log(exp(log(exp(log(x))))));

  t.push_state();
  AKS_CHECK_VARIABLE(x, 3);
  AKS_CHECK_VARIABLE(f, 3);
  vec_t<re_t> expected = {1, 0, 0};

  for (int i = 0; i < 3; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(x);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 204);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 46);
  t.pop_state();
}

void test_05() {
  std::cout << "\ntest_05" << std::endl;
  using namespace aks;
  tape_t<re_t> t;
  const var_t x = t.new_variable(1);

  var_t f = exp(exp(x));

  t.push_state();
  AKS_CHECK_VARIABLE(x, 1);
  AKS_CHECK_VARIABLE(f, 15.1542622415);
  vec_t<re_t> expected = {41.1935556747, 153.169249515,     681.502130990,
                          3478.70705883, 19853.4050763,     124537.473663,
                          848181.148608, 6213971.481006121, 48615295.31226263};

  for (int i = 0; i < 9; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(x);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 132768);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 37958);
  t.pop_state();
}

void test_06_01() {
  std::cout << "\ntest_06_01" << std::endl;
  using namespace aks;
  tape_t<re_t> t;
  const var_t x = t.new_variable(std::numbers::pi_v<re_t> / 2.0);

  var_t f = sin(x);

  t.push_state();
  AKS_CHECK_VARIABLE(x, std::numbers::pi_v<re_t> / 2.0);
  AKS_CHECK_VARIABLE(f, 1);
  vec_t<re_t> expected = {0, -1, 0, 1, 0, -1, 0, 1, 0};

  for (int i = 0; i < 9; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(x);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 140178);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 39214);
  t.pop_state();
}

void test_06_02() {
  std::cout << "\ntest_06_02" << std::endl;
  using namespace aks;
  tape_t<re_t> t;
  const var_t x = t.new_variable(std::numbers::pi_v<re_t> / 2.0);

  var_t f = cos(x);

  t.push_state();
  AKS_CHECK_VARIABLE(x, std::numbers::pi_v<re_t> / 2.0);
  AKS_CHECK_VARIABLE(f, 0);
  vec_t<re_t> expected = {-1, 0, 1, 0, -1, 0, 1, 0, -1};

  for (int i = 0; i < 9; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(x);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 183338);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 51500);
  t.pop_state();
}

void test_07() {
  std::cout << "\ntest_07" << std::endl;
  using namespace aks;
  tape_t<re_t> t;
  const var_t x = t.new_variable(2);

  var_t f =
      ((sin(x) ^ 2.0) / (log(x + 50) ^ 2.0)) * (1.0 - (exp(x) ^ (-x))) + cos(x);

  t.push_state();
  AKS_CHECK_VARIABLE(x, 2);
  AKS_CHECK_VARIABLE(f, -0.364157274408385);
  vec_t<re_t> expected = {
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
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 3728582);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 1038086);
  t.pop_state();
}

void test_08() {
  std::cout << "\ntest_08" << std::endl;
  using namespace aks;
  tape_t<re_t> t;
  const var_t x = t.new_variable(2);
  const var_t y = t.new_variable(3);

  var_t f =
      (x ^ y) * exp(x) * exp(y) / (log(y) - (y ^ -0.5)) + sin(cos(x) * y * 0.5);

  t.push_state();
  AKS_CHECK_VARIABLE(x, 2);
  AKS_CHECK_VARIABLE(y, 3);
  AKS_CHECK_VARIABLE(f, 2277.166893402295);
  vec_t<re_t> expected = {
      5693.271663432962, 12529.22628253501, 24487.49989506043,
      43265.42685712199, 70607.95619169925, 108326.8348413039,
      157615.1751647243, 218966.3917658434, 304763.4639907167};

  for (int i = 0; i < 9; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(x);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 3464018);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 970308);
  t.pop_state();
}

void test_09() {
  std::cout << "\ntest_09" << std::endl;
  using namespace aks;
  tape_t<re_t> t;
  const var_t x = t.new_variable(2);
  const var_t y = t.new_variable(3);

  var_t f = sin(y * tanh(x / 4)) / tanh(y / 6);

  t.push_state();
  AKS_CHECK_VARIABLE(x, 2);
  AKS_CHECK_VARIABLE(y, 3);
  AKS_CHECK_VARIABLE(f, 2.127248903688577);
  vec_t<re_t> expected = {
      0.234088640404172, -0.794171421323257, 0.421052063524680,
      0.403203519062006, -0.966405900729128, 0.315910699048416,
      1.943245437957466, -3.596458737095583, -1.901001142886110};

  for (int i = 0; i < 9; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(x);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 1994523);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 539981);
  t.pop_state();
}

void test_10() {
  std::cout << "\ntest_10" << std::endl;
  using namespace aks;
  tape_t<re_t> t;
  const var_t x = t.new_variable(2);
  const var_t y = t.new_variable(3);

  var_t f = (x ^ y) * exp(x) * exp(y) / (log(y) - (1.0 / sqrt(y))) +
            sin(cos(x) * y * 0.5);

  t.push_state();
  AKS_CHECK_VARIABLE(x, 2);
  AKS_CHECK_VARIABLE(y, 3);
  AKS_CHECK_VARIABLE(f, 2277.166893402295);
  vec_t<re_t> expected = {
      5693.271663432962, 12529.22628253501, 24487.49989506043,
      43265.42685712199, 70607.95619169925, 108326.8348413039,
      157615.1751647243, 218966.3917658434, 304763.4639907167};

  for (int i = 0; i < 9; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(x);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 4041640);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 1116278);
  t.pop_state();
}

void test_11() {
  std::cout << "\ntest_11" << std::endl;
  using namespace aks;
  tape_t<re_t> t;
  const var_t x = t.new_variable(0.5);

  var_t f = relu(relu(relu(x)));

  t.push_state();
  AKS_CHECK_VARIABLE(x, 0.5);
  AKS_CHECK_VARIABLE(f, 0.5);
  vec_t<re_t> expected = {1, 0, 0};

  for (int i = 0; i < 3; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(x);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 71);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 25);
  t.pop_state();
}

void test_12() {
  std::cout << "\ntest_12" << std::endl;
  using namespace aks;
  tape_t<re_t> t;
  const var_t x = t.new_variable(-0.5);

  var_t f = relu(relu(relu(x)));

  t.push_state();
  AKS_CHECK_VARIABLE(x, -0.5);
  AKS_CHECK_VARIABLE(f, 0);
  vec_t<re_t> expected = {0, 0, 0};

  for (int i = 0; i < 3; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(x);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 71);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 25);
  t.pop_state();
}

void test_13() {
  std::cout << "\ntest_13" << std::endl;
  using namespace aks;
  tape_t<re_t> t;
  const var_t x = t.new_variable(2);
  const var_t y = t.new_variable(3);

  var_t f = (x ^ y) * exp(x) * exp(y) / (log(y) - (1.0 / sqrt(y))) +
            sin(cos(x) * y * 0.5);

  f = f * f;

  t.push_state();
  AKS_CHECK_VARIABLE(x, 2);
  AKS_CHECK_VARIABLE(y, 3);
  AKS_CHECK_VARIABLE(f, 5185489.060407462);
  vec_t<re_t> expected = {
      25929059.49422991, 121888963.0483667, 539517981.8810239,
      2254246176.856630, 8920979033.396202, 33572462599.63692,
      120670307539.7625, 416043043171.6418, 1381489125066.02686};

  for (int i = 0; i < 9; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(x);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 4209458);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 1165742);
  t.pop_state();
}

void test_14() {
  std::cout << "\ntest_14" << std::endl;
  using namespace aks;
  tape_t<re_t> t;
  const var_t x = t.new_variable(0.25);
  const var_t y = t.new_variable(0.5);

  var_t f = (x ^ y) / (log(y) - (1.0 / sqrt(y))) + sin(cos(x));

  f = sqrt(f);

  t.push_state();
  AKS_CHECK_VARIABLE(x, 0.25);
  AKS_CHECK_VARIABLE(y, 0.5);
  AKS_CHECK_VARIABLE(f, 0.766163700778299);
  vec_t<re_t> expected = {-0.401093405500755, 1.843951799064022e-02,
                          -3.976997195889043, 27.94021527444540,
                          -443.0134618423613, 7768.010873937546,
                          -172271.1423942442, 4465339.597362671};

  for (int i = 0; i < 8; ++i) {
    t.zero_grad();
    backward(f);
    f = grad(x);
    AKS_CHECK_VARIABLE(f, expected[i]);
  }
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 1014346);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 268612);
  t.pop_state();
}

void test_15() {
  std::cout << "\ntest_15" << std::endl;
  using namespace aks;
  tape_t<re_t> t;
  const var_t x = t.new_variable(2);
  const var_t y = t.new_variable(3);
  const var_t z = t.new_variable(5);

  auto F = [&]() {
    return ((x + z) ^ y) * exp(0.25 * sqrt(z) * y * x) * exp(0.25 * (y + x)) /
               (log(x + y) - (1.0 / sqrt(x * y))) +
           sin(z * cos(x) * y * 0.5);
  };

  {
    t.push_state();
    var_t f = F();

    AKS_CHECK_VARIABLE(x, 2);
    AKS_CHECK_VARIABLE(y, 3);
    AKS_CHECK_VARIABLE(z, 5);
    AKS_CHECK_VARIABLE(f, 28524.51801677098);
    vec_t<re_t> expected = {60026.82047554181, 129114.1367025926,
                            278095.5325188761, 602778.7382787745,
                            1314895.486871684, 2690006.380087323,
                            5081703.820602682, 25346219.35630465};

    for (int i = 0; i < 8; ++i) {
      t.zero_grad();
      backward(f);
      f = grad(x);
      AKS_CHECK_VARIABLE(f, expected[i]);
    }
    AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 1555964);
    AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 425346);
    t.pop_state();
  }
  {
    t.push_state();
    var_t f = F();

    AKS_CHECK_VARIABLE(x, 2);
    AKS_CHECK_VARIABLE(y, 3);
    AKS_CHECK_VARIABLE(z, 5);
    AKS_CHECK_VARIABLE(f, 28524.51801677098);
    vec_t<re_t> expected = {88164.61349317656, 275673.9023954568,
                            868833.1020929292, 2754733.9246692426};

    for (int i = 0; i < 2; ++i) {
      t.zero_grad();
      backward(f);
      f = grad(y);
      AKS_CHECK_VARIABLE(f, expected[i]);
    }
    AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 431);
    AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 114);
    t.pop_state();
  }
  {
    t.push_state();
    var_t f = F();

    AKS_CHECK_VARIABLE(x, 2);
    AKS_CHECK_VARIABLE(y, 3);
    AKS_CHECK_VARIABLE(z, 5);
    AKS_CHECK_VARIABLE(f, 28524.51801677098);
    vec_t<re_t> expected = {21792.84737554386, 13945.72372321069,
                            7309.719265308730, 3064.054319595382,
                            1006.296282742556};

    for (int i = 0; i < 2; ++i) {
      t.zero_grad();
      backward(f);
      f = grad(z);
      AKS_CHECK_VARIABLE(f, expected[i]);
    }
    AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 437);
    AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 114);
    t.pop_state();
  }
}

void test_16() {
  std::cout << "\ntest_16" << std::endl;

  using namespace aks;

  tape_t<re_t> t;

  const var_t x = t.new_variable(2);
  const var_t y = t.new_variable(3);
  const var_t z = t.new_variable(4);

  auto F = [&]() {
    var_t f = (z * (x + y));
    for (int i = 0; i < 3; ++i) {
      f *= f;
    }
    return f / 10000.0;
  };

  {

    t.push_state();
    var_t f = F();

    AKS_CHECK_VARIABLE(x, 2);
    AKS_CHECK_VARIABLE(y, 3);
    AKS_CHECK_VARIABLE(z, 4);
    AKS_CHECK_VARIABLE(f, 2560000);

    vec_t<re_t> expected = {
        4096000,     5734400,     6881280,     6881280, 5505024,
        3303014.400, 1321205.760, 264241.1520, 0,       0};

    for (int i = 0; i < 4; ++i) {
      t.zero_grad();
      backward(f);
      f = grad(x);
      AKS_CHECK_VARIABLE(f, expected[i]);
    }
    AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 1993);
    AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 495);
    t.pop_state();
  }

  {

    t.push_state();
    var_t f = F();

    AKS_CHECK_VARIABLE(x, 2);
    AKS_CHECK_VARIABLE(y, 3);
    AKS_CHECK_VARIABLE(z, 4);
    AKS_CHECK_VARIABLE(f, 2560000);

    vec_t<re_t> expected = {
        4096000,     5734400,     6881280,     6881280, 5505024,
        3303014.400, 1321205.760, 264241.1520, 0,       0};

    for (int i = 0; i < 4; ++i) {
      t.zero_grad();
      backward(f);
      f = grad(y);
      AKS_CHECK_VARIABLE(f, expected[i]);
    }
    AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 2038);
    AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 508);
    t.pop_state();
  }

  {

    t.push_state();
    var_t f = F();

    AKS_CHECK_VARIABLE(x, 2);
    AKS_CHECK_VARIABLE(y, 3);
    AKS_CHECK_VARIABLE(z, 4);
    AKS_CHECK_VARIABLE(f, 2560000);

    vec_t<re_t> expected = {5120000,  8960000, 13440000, 16800000, 16800000,
                            12600000, 6300000, 1575000,  0,        0};

    for (int i = 0; i < 4; ++i) {
      t.zero_grad();
      backward(f);
      f = grad(z);
      AKS_CHECK_VARIABLE(f, expected[i]);
    }
    AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 1820);
    AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 448);
    t.pop_state();
  }
}

void test_17() {
  std::cout << "\ntest_17" << std::endl;

  using namespace aks;

  tape_t<re_t> t;

  const var_t x = t.new_variable(2);
  const var_t y = t.new_variable(3);
  const var_t z = t.new_variable(4);

  auto F = [&]() {
    var_t f = (z * (x + y));
    for (int i = 0; i < 2; ++i) {
      f *= f;
    }
    var_t f2 = (z - x) * y;
    for (int i = 0; i < 1; ++i) {
      f2 *= f2;
    }
    return f / f2;

    // return ((z * (x + y)) ^ 4.0) / (((z - x) * y) ^ 2.0);
  };

  {
    t.push_state();
    var_t f = F();

    AKS_CHECK_VARIABLE(x, 2);
    AKS_CHECK_VARIABLE(y, 3);
    AKS_CHECK_VARIABLE(z, 4);
    AKS_CHECK_VARIABLE(f, 4444.444444444444);

    vec_t<re_t> expected = {8000,   15911.11111111111, 36586.66666666666,
                            98784,  310986.6666666667, 1125040,
                            4609920};

    for (int i = 0; i < 7; ++i) {
      t.zero_grad();
      backward(f);
      f = grad(x);
      // AKS_PRINT(f);
      AKS_CHECK_VARIABLE(f, expected[i]);
    }
    AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 125917);
    AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 34521);
    t.pop_state();
  }

  {

    t.push_state();
    var_t f = F();

    AKS_CHECK_VARIABLE(x, 2);
    AKS_CHECK_VARIABLE(y, 3);
    AKS_CHECK_VARIABLE(z, 4);
    AKS_CHECK_VARIABLE(f, 4444.444444444444);

    vec_t<re_t> expected = {592.5925925925926,  355.5555555555555,
                            -252.8395061728395, 370.8312757201646,
                            -674.2386831275720, 1460.850480109739,
                            -3670.855052583448};

    for (int i = 0; i < 7; ++i) {
      t.zero_grad();
      backward(f);
      f = grad(y);
      AKS_CHECK_VARIABLE(f, expected[i]);
    }
    AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 127310);
    AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 34974);
    t.pop_state();
  }

  {

    t.push_state();
    var_t f = F();

    AKS_CHECK_VARIABLE(x, 2);
    AKS_CHECK_VARIABLE(y, 3);
    AKS_CHECK_VARIABLE(z, 4);
    AKS_CHECK_VARIABLE(f, 4444.444444444444);

    vec_t<re_t> expected = {0,      1111.111111111111,  -1666.666666666667,
                            3750,   -10416.66666666667, 34375,
                            -131250};

    for (int i = 0; i < 7; ++i) {
      t.zero_grad();
      backward(f);
      f = grad(z);
      AKS_CHECK_VARIABLE(f, expected[i]);
    }
    AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 119756);
    AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 32718);
    t.pop_state();
  }
}

void test_18() {
  std::cout << "\ntest_18" << std::endl;
  using namespace aks;
  tape_t<re_t> t;
  const var_t x = t.new_variable(2);

  var_t f =
      ((sin(x) ^ 2.0) / (log(x + 50) ^ 2.0)) * tanh((1.0 - (exp(x) ^ (-x)))) +
      cos(x);

  t.push_state();
  AKS_CHECK_VARIABLE(x, 2);
  AKS_CHECK_VARIABLE(f, -0.376226238713693);
  vec_t<re_t> expected = {-0.944550600009225, 0.344604639299095,
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
  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 1127980);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 307884);
  t.pop_state();
}

void test_19() {
  std::cout << "\ntest_19" << std::endl;
  using namespace aks;

  auto NR = [](re_t guess, auto f, re_t tolerance = 1e-6) {
    tape_t<re_t> tape;

    auto derivative = [&tape](var_t fx, var_t x) {
      tape.push_state();
      tape.zero_grad();
      backward(fx);
      var_t dfdx = grad(x);
      double r_dfdx = dfdx.value();
      tape.pop_state();
      return r_dfdx;
    };

    var_t x = tape.new_variable(guess);
    var_t fx = f(x);

    while (std::abs(fx.value()) > tolerance) {
      x = x - fx / derivative(fx, x);
      fx = f(x);
      // AKS_PRINT(x);
      // AKS_PRINT(fx);
    };

    return x.value();
  };

  AKS_CHECK_PRINT("nr01", NR(3.0, [](auto x) { return x * x - 4; }), 2.0);
  AKS_CHECK_PRINT("nr02", NR(3.0, [](auto x) { return x * x - 16; }), 4.0);
  AKS_CHECK_PRINT("nr03", NR(5.0, [](auto x) { return x * x * x - 27; }), 3.0);
  AKS_CHECK_PRINT("nr04", NR(3.0, [](auto x) { return (x ^ 4) - 16; }), 2.0);
  AKS_CHECK_PRINT("nr05", NR(0.2, [](auto x) { return sin(x); }), re_t(0.0));
  AKS_CHECK_PRINT("nr06", NR(1.2, [](auto x) { return sin(x); }),
                  std::numbers::pi_v<double>);
}

void test_20() {
  std::cout << "\ntest_20" << std::endl;
  using namespace aks;

  tape_t<re_t> t;

  var_t x = t.new_variable(2.0);
  var_t y = t.new_variable(3.0);
  var_t z = t.new_variable(5.0);

  AKS_CHECK_VARIABLE(x, 2.0);
  AKS_CHECK_VARIABLE(y, 3.0);
  AKS_CHECK_VARIABLE(z, 5.0);

  auto DIFF = [&](size_t I, size_t J, size_t K) {
    t.push_state();
    var_t f = (x * y * z) ^ 4;

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
    double result = f.value();
    t.pop_state();
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

  tape_t<re_t> t;

  var_t x = t.new_variable(2.0);
  var_t y = t.new_variable(3.0);
  var_t z = t.new_variable(5.0);

  AKS_CHECK_VARIABLE(x, 2.0);
  AKS_CHECK_VARIABLE(y, 3.0);
  AKS_CHECK_VARIABLE(z, 5.0);

  auto DIFF = [&](size_t I, size_t J, size_t K) {
    t.push_state();
    var_t f = ((z + (x * y)) ^ 4) / ((z - ((x * y) ^ 4)));
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

    double result = f.value();
    t.pop_state();
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

  tape_t<re_t> t;

  var_t x = t.new_variable(2.0);
  var_t y = t.new_variable(3.0);
  var_t z = t.new_variable(5.0);

  AKS_CHECK_VARIABLE(x, 2.0);
  AKS_CHECK_VARIABLE(y, 3.0);
  AKS_CHECK_VARIABLE(z, 5.0);

  auto DIFF = [&](size_t I, size_t J, size_t K) {
    t.push_state();
    var_t f = (z * (sin(x + y) + cos(x - y))) / (log(x) * tanh(y) * exp(z));
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

    double result = f.value();
    t.pop_state();
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

  tape_t<re_t> t;

  var_t x = t.new_variable(2.0);
  var_t y = t.new_variable(3.0);
  var_t z = t.new_variable(5.0);

  AKS_CHECK_VARIABLE(x, 2.0);
  AKS_CHECK_VARIABLE(y, 3.0);
  AKS_CHECK_VARIABLE(z, 5.0);

  auto DIFF = [&](size_t I, size_t J, size_t K) {
    t.push_state();
    var_t f = (((x * y) ^ 2.0) + ((x ^ z) - (y ^ z))) / (sqrt(z - x));
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

    double result = f.value();
    t.pop_state();
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

  tape_t<re_t> t;

  var_t x = t.new_variable(2.0);
  var_t y = t.new_variable(3.0);
  var_t z = t.new_variable(5.0);

  AKS_CHECK_VARIABLE(x, 2.0);
  AKS_CHECK_VARIABLE(y, 3.0);
  AKS_CHECK_VARIABLE(z, 5.0);

  auto DIFF = [&](size_t I, size_t J, size_t K) {
    t.push_state();
    var_t f = tanh(x * tanh(sqrt(z - x)));
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

    double result = f.value();
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

  tape_t<re_t> t;

  var_t x = t.new_variable(2.0);
  var_t y = t.new_variable(3.0);
  var_t z = t.new_variable(5.0);

  AKS_CHECK_VARIABLE(x, 2.0);
  AKS_CHECK_VARIABLE(y, 3.0);
  AKS_CHECK_VARIABLE(z, 5.0);

  auto DIFF = [&](size_t I, size_t J, size_t K) {
    t.push_state();
    var_t f = sqrt(x * exp(y - log(z)));
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

    double result = f.value();
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

  tape_t<re_t> t;

  auto to_variable = [&](re_t v) { return t.new_variable(v); };

  vec_t<var_t> xs = zipped_op(to_variable, vec_re_t{2.0, 3.0, 5.0});
  vec_t<var_t> ys = zipped_op(to_variable, vec_re_t{7.0, 11.0, 13.0});

  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 6);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 0);

  var_t d = dot(xs, ys);

  AKS_CHECK_VARIABLE(d, 112.0);

  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 7);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 0);

  t.zero_grad();
  backward(d);
  AKS_CHECK_VARIABLE(grad(xs[0]), 7.0);
  AKS_CHECK_VARIABLE(grad(ys[1]), 3.0);

  var_t s = asum(ys);
  AKS_CHECK_VARIABLE(s, 31.0);

  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 15);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 7);

  t.zero_grad();
  backward(s);
  AKS_CHECK_VARIABLE(grad(xs[0]), 0.0);
  AKS_CHECK_VARIABLE(grad(ys[1]), 1.0);

  var_t g = gsum(xs);
  AKS_CHECK_VARIABLE(g, 30.0);

  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 62);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 15);

  // backward(g);
  // AKS_CHECK_VARIABLE(grad(xs[0]), 15.0);

  var_t mx = max(xs);
  AKS_CHECK_VARIABLE(mx, 5.0);

  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 62);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 15);

  // backward(mx);
  // AKS_CHECK_VARIABLE(grad(xs[0]), 0.0);

  var_t mn = min(xs);
  AKS_CHECK_VARIABLE(mn, 2.0);

  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 62);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 15);

  // backward(mn);
  // AKS_CHECK_VARIABLE(grad(xs[0]), 1.0);

  var_t mm = mean(ys);
  AKS_CHECK_VARIABLE(mm, 10.3333333);

  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 65);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 15);

  // backward(mm);
  // AKS_CHECK_VARIABLE(grad(ys[0]), 1.0);

  var_t gm = gmean(xs);
  AKS_CHECK_VARIABLE(gm, 10.0);

  AKS_CHECK_PRINT(t.nodes_.size(), t.nodes_.size(), 68);
  AKS_CHECK_PRINT(t.grads_.size(), t.grads_.size(), 15);

  // backward(gm);
  // AKS_CHECK_VARIABLE(grad(xs[0]), 1.0);
}
} // namespace

int main() {
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
  test_02();
  test_01();

  std::cout << "\ntotal  : " << TOTAL_TEST_RUN
            << "\npassed : " << TOTAL_TEST_PASS
            << "\nfailed : " << TOTAL_TEST_FAIL << std::endl;
  return 0;
}
