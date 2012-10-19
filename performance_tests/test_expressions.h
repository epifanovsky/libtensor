#ifndef TEST_EXPRESSION_H
#define TEST_EXPRESSION_H

#include <iostream>
#include <memory>
#include <libtensor/libtensor.h>
#include <libtensor/block_tensor/btod_random.h>
#include "test_expression_i.h"

namespace libtensor {

/** \brief Simple add expression to evaluate in performance tests

      Evaluates and computes the expression
      \f[    r_{iajb} = v_{iajb} + 0.5 w_{iajb} \f]

    Use this as example howto implement expressions for performance tests

    \ingroup libtensor_performance_tests

 **/
class test_expression_simple_add :
    public test_expression_i,
    public timings<test_expression_simple_add> {

    friend class timings<test_expression_simple_add>;

private:
    typedef btensor<4,double> btensor_t;
    std::auto_ptr<btensor_t> m_v_ovov, m_w_ovov, m_res_ovov;

public:
    static const char* k_clazz;

public:
    virtual ~test_expression_simple_add() {}

    virtual void calculate() {

        test_expression_simple_add::start_timer();
        letter a, b, i, j;
        btensor_t &r_ovov(*m_res_ovov), &v_ovov(*m_v_ovov),
            &w_ovov(*m_w_ovov);

        r_ovov(i|a|j|b) = v_ovov(i|a|j|b) + 0.5 * w_ovov(i|a|j|b);
        test_expression_simple_add::stop_timer();
    }

    virtual void initialize( const bispace_data_i& bisd ) {

        test_expression_simple_add::start_timer("initialize()");
        bispace<1> i=bisd.one(), a=bisd.two();
        bispace<1> j=bisd.one(), b=bisd.two();
        bispace<4> biovov(i|a|j|b);

        btod_random<4> randr;
        m_v_ovov.reset(new btensor_t(biovov));
        randr.perform(*m_v_ovov);
        m_w_ovov.reset(new btensor_t(biovov));
        randr.perform(*m_w_ovov);
        m_res_ovov.reset(new btensor_t(biovov));
        randr.perform(*m_res_ovov);
        test_expression_simple_add::stop_timer("initialize()");
    }

};

/** \brief Permuted add expression to evaluate in performance tests

      Evaluates and computes the expression
      \f[
      r_{iajb} = 2.0 \left(ia|jb\right) - \left(ij|ab\right)
      \f]

    \ingroup libtensor_performance_tests

 **/
class test_expression_permute_add :
    public test_expression_i,
    public timings<test_expression_permute_add> {

    friend class timings<test_expression_permute_add>;

private:
    typedef btensor<4,double> btensor_t;
    std::auto_ptr<btensor_t> m_i_ovov, m_i_oovv, m_res_ovov;

public:
    static const char* k_clazz;

public:
    virtual ~test_expression_permute_add() {}

    virtual void calculate();

    virtual void initialize( const bispace_data_i& bisd );
};

/** \brief Simple copy expression to evaluate in performance tests

      Evaluates and computes the expression
      \f[
      r_{iajb} = 0.5 v_{iajb}
      \f]

    \ingroup libtensor_performance_tests

 **/
class test_expression_simple_copy :
    public test_expression_i,
    public timings<test_expression_simple_copy> {

    friend class timings<test_expression_simple_copy>;

private:
    typedef btensor<4,double> btensor_t;
    std::auto_ptr<btensor_t> m_v_ovov, m_res_ovov;

public:
    static const char* k_clazz;

public:
    virtual ~test_expression_simple_copy() {}

    virtual void calculate();

    virtual void initialize( const bispace_data_i& bisd );
};

/** \brief Permute copy expression to evaluate in performance tests

      Evaluates and computes the expression
      \f[
      r_{iajb} = 0.5 v_{ijab}
      \f]

    \ingroup libtensor_performance_tests

 **/
class test_expression_permute_copy :
    public test_expression_i,
    public timings<test_expression_permute_copy> {

    friend class timings<test_expression_permute_copy>;

private:
    typedef btensor<4,double> btensor_t;
    std::auto_ptr<btensor_t> m_v_oovv, m_res_ovov;

public:
    static const char* k_clazz;

public:
    virtual ~test_expression_permute_copy() {}

    virtual void calculate();

    virtual void initialize( const bispace_data_i& bisd );
};

/** \brief ADC type expression to evaluate in performance tests

      Evaluates and computes the expression
      \f[
      r_{iajb} = 0.5 \sum_{cd} \left(ac|bd\right) v_{icjd}
          + 0.5  \sum_{kl} \left(ik|jl\right) v_{kalb}
          - \sum_{kc} \left[
              \left(ik|ac\right) v_{kcjb}    - \left(jk|ac\right) v_{kcib}
              - \left(ik|bc\right) v_{kcja} + \left(jk|bc\right) v_{kcia}
          \right]
      \f]
      similar to an expression occuring in ADC

    \ingroup libtensor_performance_tests
 **/
class test_expression_adc :
    public test_expression_i,
    public timings<test_expression_adc> {

    friend class timings<test_expression_adc>;

private:
    typedef btensor<4,double> btensor_t;
    std::auto_ptr<btensor_t> m_v_ovov, m_res_ovov,
        m_i_vvvv, m_i_oooo, m_i_oovv;

public:
    static const char* k_clazz;

public:
    virtual ~test_expression_adc() {}

    virtual void calculate();

    virtual void initialize( const bispace_data_i& bisd );
};

} // namespace libtensor

#endif // TEST_EXPRESSION_H
