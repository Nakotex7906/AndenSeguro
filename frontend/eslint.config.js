/**
 * @file eslint.config.js
 * @description Configuración principal de ESLint para la aplicación React/TypeScript.
 * Optimizada para trabajar en conjunto con Prettier y Tailwind CSS v4.
 * ESLint se encarga estrictamente de la lógica y calidad del código,
 * mientras que Prettier asume el formateo visual y ordenamiento de clases.
 */

import js from '@eslint/js'
import globals from 'globals'
import react from 'eslint-plugin-react'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'
import query from '@tanstack/eslint-plugin-query'
import jsxA11y from 'eslint-plugin-jsx-a11y'
import simpleImportSort from 'eslint-plugin-simple-import-sort'
import eslintConfigPrettier from 'eslint-config-prettier'

export default tseslint.config(
  { ignores: ['dist', 'node_modules', '.vite'] },
  {
    extends: [
      js.configs.recommended,
      ...tseslint.configs.strictTypeChecked,
      ...tseslint.configs.stylisticTypeChecked,
    ],
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
    },
    settings: {
      react: { version: 'detect' },
    },
    plugins: {
      'react': react,
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
      '@tanstack/query': query,
      'jsx-a11y': jsxA11y,
      'simple-import-sort': simpleImportSort,
    },
    rules: {
      ...react.configs['recommended'].rules,
      ...react.configs['jsx-runtime'].rules,
      ...reactHooks.configs.recommended.rules,
      ...jsxA11y.configs.recommended.rules,
      ...query.configs['flat/recommended'][0].rules,

      /**
       * MÓDULO: Importaciones
       */
      'simple-import-sort/imports': 'error',
      'simple-import-sort/exports': 'error',

      /**
       * MÓDULO: React y Hooks
       */
      'react-refresh/only-export-components': ['warn', { allowConstantExport: true }],
      'react/prop-types': 'off',
      'react/jsx-no-leaked-render': ['error', { validStrategies: ['coerce', 'ternary'] }],
      'react-hooks/exhaustive-deps': 'error',

      /**
       * MÓDULO: TypeScript
       */
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
      '@typescript-eslint/no-explicit-any': 'error',
      '@typescript-eslint/consistent-type-imports': 'error',

      /**
       * MÓDULO: Calidad General
       */
      'no-console': ['warn', { allow: ['warn', 'error'] }],
    },
  },
  // INTEGRACIÓN: Prettier
  // Debe ir estrictamente al final. Apaga las reglas de ESLint (como stylisticTypeChecked)
  // que entran en conflicto con el formateo de Prettier.
  eslintConfigPrettier
)