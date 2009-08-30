#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_EVALFUNCTOR_CONTRACT_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_EVALFUNCTOR_CONTRACT_H

#include "core_contract.h"

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2,
	size_t NTensor1, size_t NOper1, size_t NTensor2, size_t NOper2>
class evalfunctor_contract {
public:
	static const char *k_clazz; //!< Class name
	static const size_t k_ordera = N + K; //!< Order of the first %tensor
	static const size_t k_orderb = M + K; //!< Order of the second %tensor
	static const size_t k_orderc = N + M; //!< Order of the result

	//!	Contraction expression core type
	typedef core_contract<N, M, K, T, E1, E2> core_t;

	//!	Contraction expression type
	typedef expr<k_orderc, T, core_t> expression_t;

public:
	evalfunctor_contract(
		expression_t &expr, const letter_expr<k_orderc> &label) {

	}

	void evaluate() throw(exception);

};


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
class evalfunctor_contract<N, M, K, T, E1, E2, 1, 0, 1, 0> {
public:
	static const char *k_clazz; //!< Class name
	static const size_t k_ordera = N + K; //!< Order of the first %tensor
	static const size_t k_orderb = M + K; //!< Order of the second %tensor
	static const size_t k_orderc = N + M; //!< Order of the result

	//!	Contraction expression core type
	typedef core_contract<N, M, K, T, E1, E2> core_t;

	//!	Contraction expression type
	typedef expr<k_orderc, T, core_t> expression_t;

private:

public:
	evalfunctor_contract(
		expression_t &expr, const letter_expr<k_orderc> &label) { }
	void evaluate() throw(exception);

};


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2,
	size_t NTensor1, size_t NOper1, size_t NTensor2, size_t NOper2>
const char *evalfunctor_contract<N, M, K, T, E1, E2,
	NTensor1, NOper1, NTensor2, NOper2>::k_clazz =
		"evalfunctor_contract<N, M, K, T, E1, E2, "
		"NTensor1, NOper1, NTensor2, NOper2>";


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2,
	size_t NTensor1, size_t NOper1, size_t NTensor2, size_t NOper2>
void evalfunctor_contract<N, M, K, T, E1, E2,
	NTensor1, NOper1, NTensor2, NOper2>::evaluate() throw(exception) {

	throw_exc(k_clazz, "evaluate()", "NIY");
}


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
const char *evalfunctor_contract<N, M, K, T, E1, E2, 1, 0, 1, 0>::k_clazz =
		"evalfunctor_contract<N, M, K, T, E1, E2, 1, 0, 1, 0>";


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
void evalfunctor_contract<N, M, K, T, E1, E2, 1, 0, 1, 0>::evaluate()
	throw(exception) {

	throw_exc(k_clazz, "evaluate()", "NIY");
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_EVALFUNCTOR_CONTRACT_H
