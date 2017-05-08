/*******************************************************************\

Module: C++ Language Conversion

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_CPP_CONVERT_CHARACTER_LITERAL_H
#define CPROVER_CPP_CONVERT_CHARACTER_LITERAL_H

#include <string>
#include <expr.h>

void convert_character_literal(const std::string &src, exprt &dest);

#endif
