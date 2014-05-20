/*
 * range.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: smanzer
 */

#include "range.h"

namespace libtensor
{

block_list range(size_t min,size_t max)
{
	block_list the_range;
	for(size_t i = min; i < max; ++i)
	{
		the_range.push_back(i);
	}
	return the_range;
}

} /* namespace libtensor */
