/*******************************************************************\

Module: Main Module

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk
		Jeremy Morse, jcmm106@ecs.soton.ac.uk

\*******************************************************************/

/*

  ESBMC
  SMT-based Context-Bounded Model Checking for ANSI-C/C++
  Copyright (c) 2009-2011, Lucas Cordeiro, Federal University of Amazonas
  Jeremy Morse, Denis Nicole, Bernd Fischer, University of Southampton,
  Joao Marques Silva, University College Dublin.
  All rights reserved.

*/

#include <stdint.h>

#include <irep2.h>

#include <langapi/mode.h>

#include "parseoptions.h"

/*******************************************************************\

Function: main

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void dereference_handlers_init(void);

int main(int argc, const char **argv)
{
  // To avoid the static initialization order fiasco:
  type_poolt bees(true);
  type_pool = bees;
  init_expr_constants();
  dereference_handlers_init();

  cbmc_parseoptionst parseoptions(argc, argv);
  return parseoptions.main();
}

const mode_table_et mode_table[] =
{
  LANGAPI_HAVE_MODE_C,
  LANGAPI_HAVE_MODE_CPP,
#ifdef USE_SPECC
  LANGAPI_HAVE_MODE_SPECC,
#endif
#ifdef USE_PHP
  LANGAPI_HAVE_MODE_PHP,
#endif
  LANGAPI_HAVE_MODE_END
};

extern "C" uint8_t buildidstring_buf[1];
uint8_t *version_string = buildidstring_buf;
