#ifndef LIBTENSOR_EXPR_MUL_FUNCTOR_H
#define LIBTENSOR_EXPR_MUL_FUNCTOR_H

namespace libtensor {

/**	\brief Multiplication operation functor

	\ingroup libtensor_expressions
**/
template<typename T1, typename T2>
class expr_mul_functor {
public:
	static T1 eval(T1 &a, T2 &b);
};

} // namespace libtensor

#endif // LIBTENSOR_EXPR_MUL_FUNCTOR_H

