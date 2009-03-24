#ifndef LIBTENSOR_EXPR_OP_SUB_H
#define LIBTENSOR_EXPR_OP_SUB_H

namespace libtensor {

/**	\brief Binary operation: substraction

	\ingroup libtensor_expressions
**/
template<typename T1, typename T2>
class expr_op_sub {
public:
	static T1 eval(T1 a, T2 b);
};

} // namespace libtensor

#endif // LIBTENSOR_EXPR_OP_SUB_H

