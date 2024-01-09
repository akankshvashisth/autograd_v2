
#include "autograd_v2.hpp"

namespace {
constexpr bool QUIET_PASS = true;
static size_t TOTAL_TEST_RUN = 0;

#define AKS_PRINT(EXPR)                                                        \
  std::cout << std::setprecision(15) << #EXPR << " = " << EXPR << std::endl

#define AKS_PRINT_AS(NAME, EXPR)                                               \
  std::cout << std::setprecision(15) << NAME << " " << EXPR << "       \t("    \
            << #EXPR << ")" << std::endl

#define AKS_CHECK_PRINT(EXPR, EXPR_VAL, EXPECTED)                              \
  do {                                                                         \
    ++TOTAL_TEST_RUN;                                                          \
    const double diff = std::abs(double(EXPR_VAL) - double(EXPECTED));         \
    if (std::isnan(double(EXPR_VAL)) || diff > 1e-4) {                         \
      std::cout << std::setprecision(18) << "\nCHECK FAILED: " << #EXPR        \
                << " = " << double(EXPR_VAL) << " != " << EXPECTED << " ("     \
                << diff << ")"                                                 \
                << " on line " << __LINE__ << " in " << __FILE__ << std::endl; \
      assert(true);                                                            \
    } else {                                                                   \
      if (!QUIET_PASS) {                                                       \
        std::cout << std::setprecision(15) << "____pass____: " << #EXPR        \
                  << " = " << double(EXPR_VAL) << std::endl;                   \
      } else {                                                                 \
        std::cout << ".";                                                      \
      }                                                                        \
    }                                                                          \
  } while (false)

#define AKS_CHECK_VARIABLE(EXPR, EXPECTED)                                     \
  AKS_CHECK_PRINT(EXPR, EXPR.value(), EXPECTED)
} // namespace

namespace {

void test_01() {
  std::cout << "\ntest_01" << std::endl;
  using namespace aks;

  tape_t t;

  const variable x = t.new_variable(3.0);
  variable y = t.new_variable(5.0);

  variable f = (x * x * x * x * x * x * x * x);

  t.push_state();
  AKS_CHECK_VARIABLE(x, 3);
  AKS_CHECK_VARIABLE(y, 5);
  AKS_CHECK_VARIABLE(f, 6561);
  vec_t<real_t> expected = {17496,  40824,  81648, 136080, 181440,
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
  tape_t t;
  const variable x = t.new_variable(3.0);
  const variable y = t.new_variable(8.0);

  variable f = pow(x, y);

  t.push_state();
  AKS_CHECK_VARIABLE(x, 3);
  AKS_CHECK_VARIABLE(y, 8);
  AKS_CHECK_VARIABLE(f, 6561);
  vec_t<real_t> expected = {17496,  40824,  81648, 136080, 181440,
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
  tape_t t;
  const variable x = t.new_variable(3.0);

  variable f = pow(x, 8.0);

  t.push_state();
  AKS_CHECK_VARIABLE(x, 3);
  AKS_CHECK_VARIABLE(f, 6561);
  vec_t<real_t> expected = {17496,  40824,  81648, 136080, 181440,
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
  tape_t t;
  const variable x = t.new_variable(3.0);

  variable f = exp(log(exp(log(exp(log(x))))));

  t.push_state();
  AKS_CHECK_VARIABLE(x, 3);
  AKS_CHECK_VARIABLE(f, 3);
  vec_t<real_t> expected = {1, 0, 0};

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
  tape_t t;
  const variable x = t.new_variable(1);

  variable f = exp(exp(x));

  t.push_state();
  AKS_CHECK_VARIABLE(x, 1);
  AKS_CHECK_VARIABLE(f, 15.1542622415);
  vec_t<real_t> expected = {
      41.1935556747, 153.169249515,     681.502130990,
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

void test_06() {
  std::cout << "\ntest_06" << std::endl;
  using namespace aks;
  tape_t t;
  const variable x = t.new_variable(std::numbers::pi_v<real_t> / 2.0);

  variable f = sin(x);

  t.push_state();
  AKS_CHECK_VARIABLE(x, std::numbers::pi_v<real_t> / 2.0);
  AKS_CHECK_VARIABLE(f, 1);
  vec_t<real_t> expected = {0, -1, 0, 1, 0, -1, 0, 1, 0};

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

void test_07() {
  std::cout << "\ntest_07" << std::endl;
  using namespace aks;
  tape_t t;
  const variable x = t.new_variable(2);

  variable f =
      ((sin(x) ^ 2.0) / (log(x + 50) ^ 2.0)) * (1.0 - (exp(x) ^ (-x))) + cos(x);

  t.push_state();
  AKS_CHECK_VARIABLE(x, 2);
  AKS_CHECK_VARIABLE(f, -0.364157274408385);
  vec_t<real_t> expected = {
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
  tape_t t;
  const variable x = t.new_variable(2);
  const variable y = t.new_variable(3);

  variable f =
      (x ^ y) * exp(x) * exp(y) / (log(y) - (y ^ -0.5)) + sin(cos(x) * y * 0.5);

  t.push_state();
  AKS_CHECK_VARIABLE(x, 2);
  AKS_CHECK_VARIABLE(y, 3);
  AKS_CHECK_VARIABLE(f, 2277.166893402295);
  vec_t<real_t> expected = {
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
  tape_t t;
  const variable x = t.new_variable(2);
  const variable y = t.new_variable(3);

  variable f = sin(y * tanh(x / 4)) / tanh(y / 6);

  t.push_state();
  AKS_CHECK_VARIABLE(x, 2);
  AKS_CHECK_VARIABLE(y, 3);
  AKS_CHECK_VARIABLE(f, 2.127248903688577);
  vec_t<real_t> expected = {
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
  tape_t t;
  const variable x = t.new_variable(2);
  const variable y = t.new_variable(3);

  variable f = (x ^ y) * exp(x) * exp(y) / (log(y) - (1.0 / sqrt(y))) +
               sin(cos(x) * y * 0.5);

  t.push_state();
  AKS_CHECK_VARIABLE(x, 2);
  AKS_CHECK_VARIABLE(y, 3);
  AKS_CHECK_VARIABLE(f, 2277.166893402295);
  vec_t<real_t> expected = {
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
  tape_t t;
  const variable x = t.new_variable(0.5);

  variable f = relu(relu(relu(x)));

  t.push_state();
  AKS_CHECK_VARIABLE(x, 0.5);
  AKS_CHECK_VARIABLE(f, 0.5);
  vec_t<real_t> expected = {1, 0, 0};

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
  tape_t t;
  const variable x = t.new_variable(-0.5);

  variable f = relu(relu(relu(x)));

  t.push_state();
  AKS_CHECK_VARIABLE(x, -0.5);
  AKS_CHECK_VARIABLE(f, 0);
  vec_t<real_t> expected = {0, 0, 0};

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
  tape_t t;
  const variable x = t.new_variable(2);
  const variable y = t.new_variable(3);

  variable f = (x ^ y) * exp(x) * exp(y) / (log(y) - (1.0 / sqrt(y))) +
               sin(cos(x) * y * 0.5);

  f = f * f;

  t.push_state();
  AKS_CHECK_VARIABLE(x, 2);
  AKS_CHECK_VARIABLE(y, 3);
  AKS_CHECK_VARIABLE(f, 5185489.060407462);
  vec_t<real_t> expected = {
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
  tape_t t;
  const variable x = t.new_variable(0.25);
  const variable y = t.new_variable(0.5);

  variable f = (x ^ y) / (log(y) - (1.0 / sqrt(y))) + sin(cos(x));

  f = sqrt(f);

  t.push_state();
  AKS_CHECK_VARIABLE(x, 0.25);
  AKS_CHECK_VARIABLE(y, 0.5);
  AKS_CHECK_VARIABLE(f, 0.766163700778299);
  vec_t<real_t> expected = {-0.401093405500755, 1.843951799064022e-02,
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
  tape_t t;
  const variable x = t.new_variable(2);
  const variable y = t.new_variable(3);
  const variable z = t.new_variable(5);

  auto F = [&]() {
    return ((x + z) ^ y) * exp(0.25 * sqrt(z) * y * x) * exp(0.25 * (y + x)) /
               (log(x + y) - (1.0 / sqrt(x * y))) +
           sin(z * cos(x) * y * 0.5);
  };

  {
    t.push_state();
    variable f = F();

    AKS_CHECK_VARIABLE(x, 2);
    AKS_CHECK_VARIABLE(y, 3);
    AKS_CHECK_VARIABLE(z, 5);
    AKS_CHECK_VARIABLE(f, 28524.51801677098);
    vec_t<real_t> expected = {60026.82047554181, 129114.1367025926,
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
    variable f = F();

    AKS_CHECK_VARIABLE(x, 2);
    AKS_CHECK_VARIABLE(y, 3);
    AKS_CHECK_VARIABLE(z, 5);
    AKS_CHECK_VARIABLE(f, 28524.51801677098);
    vec_t<real_t> expected = {88164.61349317656, 275673.9023954568,
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
    variable f = F();

    AKS_CHECK_VARIABLE(x, 2);
    AKS_CHECK_VARIABLE(y, 3);
    AKS_CHECK_VARIABLE(z, 5);
    AKS_CHECK_VARIABLE(f, 28524.51801677098);
    vec_t<real_t> expected = {21792.84737554386, 13945.72372321069,
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

  tape_t t;

  const variable x = t.new_variable(2);
  const variable y = t.new_variable(3);
  const variable z = t.new_variable(4);

  auto F = [&]() {
    variable f = (z * (x + y));
    for (int i = 0; i < 3; ++i) {
      f *= f;
    }
    return f / 10000.0;
  };

  {

    t.push_state();
    variable f = F();

    AKS_CHECK_VARIABLE(x, 2);
    AKS_CHECK_VARIABLE(y, 3);
    AKS_CHECK_VARIABLE(z, 4);
    AKS_CHECK_VARIABLE(f, 2560000);

    vec_t<real_t> expected = {
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
    variable f = F();

    AKS_CHECK_VARIABLE(x, 2);
    AKS_CHECK_VARIABLE(y, 3);
    AKS_CHECK_VARIABLE(z, 4);
    AKS_CHECK_VARIABLE(f, 2560000);

    vec_t<real_t> expected = {
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
    variable f = F();

    AKS_CHECK_VARIABLE(x, 2);
    AKS_CHECK_VARIABLE(y, 3);
    AKS_CHECK_VARIABLE(z, 4);
    AKS_CHECK_VARIABLE(f, 2560000);

    vec_t<real_t> expected = {5120000,  8960000, 13440000, 16800000, 16800000,
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

  tape_t t;

  const variable x = t.new_variable(2);
  const variable y = t.new_variable(3);
  const variable z = t.new_variable(4);

  auto F = [&]() {
    variable f = (z * (x + y));
    for (int i = 0; i < 2; ++i) {
      f *= f;
    }
    variable f2 = (z - x) * y;
    for (int i = 0; i < 1; ++i) {
      f2 *= f2;
    }
    return f / f2;

    // return ((z * (x + y)) ^ 4.0) / (((z - x) * y) ^ 2.0);
  };

  {
    t.push_state();
    variable f = F();

    AKS_CHECK_VARIABLE(x, 2);
    AKS_CHECK_VARIABLE(y, 3);
    AKS_CHECK_VARIABLE(z, 4);
    AKS_CHECK_VARIABLE(f, 4444.444444444444);

    vec_t<real_t> expected = {8000,   15911.11111111111, 36586.66666666666,
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
    variable f = F();

    AKS_CHECK_VARIABLE(x, 2);
    AKS_CHECK_VARIABLE(y, 3);
    AKS_CHECK_VARIABLE(z, 4);
    AKS_CHECK_VARIABLE(f, 4444.444444444444);

    vec_t<real_t> expected = {592.5925925925926,  355.5555555555555,
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
    variable f = F();

    AKS_CHECK_VARIABLE(x, 2);
    AKS_CHECK_VARIABLE(y, 3);
    AKS_CHECK_VARIABLE(z, 4);
    AKS_CHECK_VARIABLE(f, 4444.444444444444);

    vec_t<real_t> expected = {0,      1111.111111111111,  -1666.666666666667,
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
  tape_t t;
  const variable x = t.new_variable(2);

  variable f =
      ((sin(x) ^ 2.0) / (log(x + 50) ^ 2.0)) * tanh((1.0 - (exp(x) ^ (-x)))) +
      cos(x);

  t.push_state();
  AKS_CHECK_VARIABLE(x, 2);
  AKS_CHECK_VARIABLE(f, -0.376226238713693);
  vec_t<real_t> expected = {-0.944550600009225, 0.344604639299095,
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

} // namespace

int main() {
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
  test_06();
  test_05();
  test_04();
  test_03();
  test_02();
  test_01();

  std::cout << "\nDONE: " << TOTAL_TEST_RUN << " checks" << std::endl;
  return 0;
}
