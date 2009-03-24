#ifndef LIBTENSOR_EXPR_ADD_FUNCTOR_H
#define LIBTENSOR_EXPR_ADD_FUNCTOR_H

namespace libtensor {

/**	\brief Addition operation functor

	\ingroup libtensor_expressions
**/
template<typename TA, typename TB>
class expr_add_functor {
public:
	static void eval(TA a, TB b);
};

template<typename T1, typename T2>
inline expr< expr_binary< expr<T1>, expr<T2>, expr_add_functor<T1,T2> > >
	operator+(const expr<T1> &a, const expr<T2> &b) {
	typedef expr_binary< expr<T1>, expr<T2>, expr_add_functor<T1,T2> >
		expr_t;
	return expr<expr_t>(expr_t(a,b));
}

} // namespace libtensor

#endif // LIBTENSOR_EXPR_ADD_FUNCTOR_H

