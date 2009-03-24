#ifndef LIBTENSOR_EXPR_SUB_FUNCTOR_H
#define LIBTENSOR_EXPR_SUB_FUNCTOR_H

namespace libtensor {

/**	\brief Substraction operation functor

	\ingroup libtensor_expressions
**/
template<typename TA, typename TB>
class expr_sub_functor {
public:
	static void eval(TA &a, TB &b);
};

} // namespace libtensor

#endif // LIBTENSOR_EXPR_SUB_FUNCTOR_H

