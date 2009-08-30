#include "test_expressions.h"

namespace libtensor {

const char* test_expression_simple_add::k_clazz="test_expression_simple_add";
const char* test_expression_permute_add::k_clazz="test_expression_permute_add";
const char* test_expression_simple_copy::k_clazz="test_expression_simple_copy";
const char* test_expression_permute_copy::k_clazz="test_expression_permute_copy";
const char* test_expression_adc::k_clazz="test_expression_adc";

void test_expression_permute_add::calculate() {
	test_expression_permute_add::start_timer();

	letter a, b, i, j;
	btensor_t &r_ovov(*m_res_ovov), &i_oovv(*m_i_oovv),	&i_ovov(*m_i_ovov);

	r_ovov(i|a|j|b) = 2.0*i_ovov(i|a|j|b) - i_oovv(i|j|a|b);

	test_expression_permute_add::stop_timer();
}

void test_expression_permute_add::initialize( const bispace_data_i& bisd ) {
	test_expression_permute_add::start_timer("initialize()");
	bispace<1> i=bisd.one(), a=bisd.two();
	bispace<1> j=bisd.one(), b=bisd.two();
	bispace<4> biovov(i*a*j*b);
	bispace<4> bioovv(i*j*a*b);

	btod_random<4> randr;
	m_res_ovov.reset(new btensor_t(biovov));
	randr.perform(*m_res_ovov);
	m_i_ovov.reset(new btensor_t(biovov));
	randr.perform(*m_i_ovov);
	m_i_oovv.reset(new btensor_t(bioovv));
	randr.perform(*m_i_oovv);

	test_expression_permute_add::stop_timer("initialize()");
}

void test_expression_simple_copy::calculate() {

	test_expression_simple_copy::start_timer();

	letter a, b, i, j;
	btensor_t &r_ovov(*m_res_ovov), &v_ovov(*m_v_ovov);

	r_ovov(i|a|j|b) = 0.5 * v_ovov(i|a|j|b);

	test_expression_simple_copy::stop_timer();
}

void test_expression_simple_copy::initialize( const bispace_data_i& bisd ) {
	test_expression_simple_copy::start_timer("initialize()");

	bispace<1> i=bisd.one(), a=bisd.two();
	bispace<1> j=bisd.one(), b=bisd.two();
	bispace<4> biovov(i*a*j*b);

	btod_random<4> randr;
	m_v_ovov.reset(new btensor_t(biovov));
	randr.perform(*m_v_ovov);
	m_res_ovov.reset(new btensor_t(biovov));
	randr.perform(*m_res_ovov);

	test_expression_simple_copy::stop_timer("initialize()");
}

void test_expression_permute_copy::calculate() {
	test_expression_permute_copy::start_timer();

	letter a, b, i, j;
	btensor_t &r_ovov(*m_res_ovov), &v_oovv(*m_v_oovv);

	r_ovov(i|a|j|b) = 0.5 * v_oovv(i|j|a|b);

	test_expression_permute_copy::stop_timer();
}

void test_expression_permute_copy::initialize( const bispace_data_i& bisd ) {
	test_expression_permute_copy::start_timer("initialize()");

	bispace<1> i=bisd.one(), a=bisd.two();
	bispace<1> j=bisd.one(), b=bisd.two();
	bispace<4> biovov(i*a*j*b);
	bispace<4> bioovv(i*j*a*b);

	btod_random<4> randr;
	m_v_oovv.reset(new btensor_t(bioovv));
	randr.perform(*m_v_oovv);
	m_res_ovov.reset(new btensor_t(biovov));
	randr.perform(*m_res_ovov);

	test_expression_permute_copy::stop_timer("initialize()");
}

void test_expression_adc::calculate() {
	test_expression_adc::start_timer();

	letter a, b, c, d, i, j, k, l;
	btensor_t &r_ovov(*m_res_ovov), &v_ovov(*m_v_ovov), &i_oovv(*m_i_oovv),
		&i_oooo(*m_i_oooo), &i_vvvv(*m_i_vvvv);

	r_ovov(i|a|j|b) = 0.5*contract(c|d, i_vvvv(a|c|b|d), v_ovov(i|c|j|d) )
		+ 0.5*contract(k|l, i_oooo(i|k|j|l), v_ovov(k|a|l|b))
		- contract(k|c, i_oovv(i|k|a|c), v_ovov(k|c|j|b))
		+ contract(k|c, i_oovv(j|k|a|c), v_ovov(k|c|i|b))
		+ contract(k|c, i_oovv(i|k|b|c), v_ovov(k|c|j|a))
		- contract(k|c, i_oovv(j|k|b|c), v_ovov(k|c|i|a));

	test_expression_adc::stop_timer();
}

void test_expression_adc::initialize( const bispace_data_i& bisd ) {
	test_expression_adc::start_timer("initialize()");

	bispace<1> i=bisd.one(), a=bisd.two(), j=bisd.one(), b=bisd.two();
	bispace<1> k=bisd.one(), l=bisd.one(), c=bisd.two(), d=bisd.two();
	bispace<4> biovov(i*a*j*b, (i*j)*(a*b));
	bispace<4> bioovv(i*j*a*b, (i*j)*(a*b));
	bispace<4> bioooo(i*j*k*l, (i*j)*(k*l));
	bispace<4> bivvvv(a*b*c*d, (a*b)*(c*d));

	btod_random<4> randr;
	m_v_ovov.reset(new btensor_t(biovov));
	randr.perform(*m_v_ovov);
	m_res_ovov.reset(new btensor_t(biovov));
	randr.perform(*m_res_ovov);
	m_i_oovv.reset(new btensor_t(bioovv));
	randr.perform(*m_i_oovv);
	m_i_oooo.reset(new btensor_t(bioooo));
	randr.perform(*m_i_oooo);
	m_i_vvvv.reset(new btensor_t(bivvvv));
	randr.perform(*m_i_vvvv);

	test_expression_adc::stop_timer("initialize()");
}


}
