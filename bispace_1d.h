#ifndef LIBTENSOR_BISPACE_1D_H
#define	LIBTENSOR_BISPACE_1D_H

#include "defs.h"
#include "exception.h"
#include "bispace_i.h"
#include "dimensions.h"
#include "bispace.h"

namespace libtensor {

/**	\brief One-dimensional block %index space

	\ingroup libtensor
 **/
class bispace_1d : public bispace_i < 1 > {
private:
	dimensions < 1 > m_dims; //!< Space %dimensions

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates the block %index space with a given dimension
	 **/
	bispace_1d(size_t dim);

	/**	\brief Virtual destructor
	 **/
	virtual ~bispace_1d();

	//@}

	//!	\name Implementation of bispace_i<1>
	//@{

	virtual rc_ptr<bispace_i < 1 > > clone() const;

	//@}

	//!	\name Implementation of ispace_i<1>
	//@{

	virtual const dimensions < 1 > &dims() const;

	//@}

private:
	/**	\brief Private constructor for cloning
	 **/
	bispace_1d(const dimensions < 1 > &dims);

	static dimensions < 1 > make_dims(size_t sz);
};

inline bispace_1d::bispace_1d(size_t dim) : m_dims(make_dims(dim)) {
}

inline bispace_1d::bispace_1d(const dimensions < 1 > &dims) : m_dims(dims) {
}

inline bispace_1d::~bispace_1d() {
}

inline rc_ptr<bispace_i < 1 > > bispace_1d::clone() const {
	return rc_ptr<bispace_i < 1 > >(new bispace_1d(m_dims));
}

inline const dimensions < 1 > &bispace_1d::dims() const {
	return m_dims;
}

inline dimensions < 1 > bispace_1d::make_dims(size_t sz) {
	index < 1 > i1, i2;
	i2[0] = sz - 1;
	index_range < 1 > ir(i1, i2);
	return dimensions < 1 > (ir);
}

} // namespace libtensor

#endif // LIBTENSOR_BISPACE_1D_H

