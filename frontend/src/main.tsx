/**
 * @fileoverview Punto de entrada principal de la aplicación React.
 * Inicializa la aplicación y la monta en el DOM de forma segura.
 */

import './index.css';

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import App from './App.tsx';

/**
 * Contenedor principal del DOM donde se inyectará la aplicación React.
 * @type {HTMLElement | null}
 */
const rootElement = document.getElementById('root');

// Manejo seguro del DOM: Verificamos que el elemento exista antes de renderizar
if (!rootElement) {
  throw new Error(
    "Error de inicialización: No se encontró el elemento con ID 'root' en el DOM. Verifica el archivo index.html."
  );
}

const root = createRoot(rootElement);

root.render(
  <StrictMode>
    <App />
  </StrictMode>
);