#ifndef TEST_EXPRESSION_H
#define TEST_EXPRESSION_H

#include <memory>
#include <libtensor.h>

#include "bispace_data.h"

namespace libtensor {
	
	
/** \brief Base class for representing an expression
  
	Base class for expression evaluation in expression performance tests. 
	Each derived class must implement the two functions:
	\li initialize( const bispace_data_i<N>& bispaces ) which initializes all 
		necessary block tensors required in the calculation based on the 
		information in bispaces	   
	\li calculate() performs the evaluation and calculation of the expression.
	
	\ingroup libtensor_performance_tests
 **/
class test_expression_i 
{
public:
	virtual void initialize( const bispace_data_i& bispaces ) = 0;
	virtual void calculate() = 0;
}; 


/** \brief Add expression to evaluate in performance tests  
  
  	Evaluates and computes the expression
  	\f[
  	r_{iajb} = v_{iajb} + 0.5 \left<ij||ab\right> 
  		- 2.0 v_{ibja} - \left<ij||ba\right> + 1.5 \left<ib||ja\right>
  	\f]
  
 	Use this as example howto implement expressions for performance tests
 **/
class test_expression_add : public test_expression_i
{
	typedef btensor<4,double> btensor_t;
	std::auto_ptr<btensor_t> m_v_ovov, m_res_ovov, 
		m_i_oovv, m_i_ovov;
		 
public:
	virtual ~test_expression_add()	{}
	virtual void calculate() {
		letter a, b, c, d, i, j, k, l;
		btensor_t &r_ovov(*m_res_ovov), &v_ovov(*m_v_ovov), &i_oovv(*m_i_oovv),
			&i_ovov(*m_i_ovov);
		
		r_ovov(i|a|j|b) = v_ovov(i|a|j|b) + 0.5 * i_oovv(i|j|a|b) 
			- 2.0 * v_ovov(i|b|j|a) - i_oovv(i|j|b|a) 
			+ 1.5 * i_ovov(i|b|j|a);
	} 	
	
	virtual void initialize( const bispace_data_i& bisd ) {
		bispace<1> bio=bisd.one(), biv=bisd.two();
		bispace<4> biovov(bio*biv*bio*biv, (bio&bio)*(biv&biv));
		bispace<4> bioovv(bio*bio*biv*biv, (bio&bio)*(biv&biv));
		
		btod_random<4> randr;
		m_v_ovov.reset(new btensor_t(biovov));
		randr.perform(*m_v_ovov);
		m_res_ovov.reset(new btensor_t(biovov));
		randr.perform(*m_res_ovov);
		m_i_ovov.reset(new btensor_t(biovov));
		randr.perform(*m_i_ovov);
		m_i_oovv.reset(new btensor_t(bioovv));
		randr.perform(*m_i_oovv);
	}
	
};


/** \brief ADC expression to evaluate in performance tests  
  
  	Evaluates and computes the expression 
  	\f[
  	r_{iajb} = 0.5 \sum_{cd} \left(ac|bd\right) v_{icjd} 
  		+ 0.5  \sum_{kl} \left(ik|jl\right) v_{kalb}
  		- \sum_{kc} \left[
  			\left(ik|ac\right) v_{kcjb}	- \left(jk|ac\right) v_{kcib}
  			- \left(ik|bc\right) v_{kcja} + \left(jk|bc\right) v_{kcia}
  		\right]
  	\f]
  	similar to an expression occuring in ADC
  
 	Use this as example howto implement expressions for performance tests
 **/

class test_expression_adc : public test_expression_i
{
	typedef btensor<4,double> btensor_t;
	std::auto_ptr<btensor_t> m_v_ovov, m_res_ovov, 
		m_i_vvvv, m_i_oooo, m_i_oovv;
		 
public:
	virtual ~test_expression_adc() {}
	virtual void calculate() {
		letter a, b, c, d, i, j, k, l;
		btensor_t &r_ovov(*m_res_ovov), &v_ovov(*m_v_ovov), &i_oovv(*m_i_oovv),
			&i_oooo(*m_i_oooo), &i_vvvv(*m_i_vvvv);
		
		r_ovov(i|a|j|b) = 0.5*contract(c|d, i_vvvv(a|c|b|d), v_ovov(i|c|j|d) )  
			+ 0.5*contract(k|l, i_oooo(i|k|j|l), v_ovov(k|a|l|b)) 
			- contract(k|c, i_oovv(i|k|a|c), v_ovov(k|c|j|b)) 
			+ contract(k|c, i_oovv(j|k|a|c), v_ovov(k|c|i|b)) 
			+ contract(k|c, i_oovv(i|k|b|c), v_ovov(k|c|j|a)) 
			- contract(k|c, i_oovv(j|k|b|c), v_ovov(k|c|i|a));
	} 	
	
	virtual void initialize( const bispace_data_i& bisd ) {
		bispace<1> bio=bisd.one(), biv=bisd.two();
		bispace<4> biovov(bio*biv*bio*biv, (bio&bio)*(biv&biv));
		bispace<4> bioovv(bio*bio*biv*biv, (bio&bio)*(biv&biv));
		bispace<4> bioooo(bio*bio*bio*bio, (bio&bio)&(bio&bio));
		bispace<4> bivvvv(biv*biv*biv*biv, (biv&biv)&(biv&biv));
		
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
	}
	
};

} // namespace libtensor

#endif // TEST_EXPRESSION_H 
