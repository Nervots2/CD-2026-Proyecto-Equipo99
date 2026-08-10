# CD-2026-Proyecto-Equipo99

Proyecto Taller Ciencia de Datos 01

Contexto ficticio: Plataforma de videojuegos "NexoPlay"

¿Cuál es el problema?
NexoPlay ha detectado un aumento del 18% en la tasa de cancelación de suscripciones (Churn) durante el último trimestre. Esto genera una pérdida crítica de ingresos recurrentes y encarece el costo de adquisición de nuevos usuarios, ya que es más caro conseguir un cliente nuevo que retener a uno existente.

¿Quién toma decisiones?
La Directora de Retención y Fidelización (apoyada por el equipo de Marketing), quien definirá las estrategias comerciales, promociones y correos de re-enganche basándose en los resultados del análisis.

¿Qué impacto tendría resolverlo?
Desarrollar un modelo predictivo permitirá identificar proactivamente qué usuarios están en riesgo de abandonar la plataforma. Esto optimizará el presupuesto de marketing al dirigir ofertas de retención solo a quienes realmente lo necesitan, estabilizando los ingresos y aumentando el valor del ciclo de vida del cliente (LTV).

Pregunta 1
¿Qué patrones de comportamiento de juego y nivel de actividad en la plataforma influyen significativamente en la probabilidad de que un usuario cancele su suscripción en el próximo trimestre?

Pregunta 2
¿Existen indicadores transaccionales o de soporte técnico que permitan predecir de manera temprana el riesgo de abandono de un cliente antes de que solicite la baja?

| Fase | ¿Cómo se aplicaría en su proyecto? |
| :--- | :--- |
| **Business Understanding** | Definición del problema del incremento del $18\%$ en el *Churn* de la plataforma NexoPlay, estableciendo como objetivo comercial reducir la pérdida de ingresos recurrentes y optimizar el presupuesto de marketing mediante un modelo predictivo que identifique a los usuarios en riesgo de abandono. |
| **Data Understanding** | Recopilación y exploración inicial de las fuentes de datos disponibles en NexoPlay, tales como registros de actividad de juego, historial de transacciones, métodos de pago, historial de soporte técnico y métricas de uso de la plataforma para entender su estructura y calidad. |
| **Data Preparation** | Limpieza y transformación de los datos, incluyendo el tratamiento de valores nulos, la creación de variables derivadas (como la frecuencia de inicio de sesión o caídas abruptas de actividad), y la partición del conjunto de datos en entrenamiento y prueba. |
| **Modeling** | Selección, entrenamiento y ajuste de hiperparámetros de algoritmos de clasificación de *Machine Learning* (como Regresión Logística, Árboles de Decisión o XGBoost) para estimar la probabilidad de cancelación de suscripción de cada usuario. |
| **Evaluation** | Evaluación del desempeño técnico de los modelos utilizando métricas de rendimiento (como AUC-ROC, *Precision*, *Recall* y *F1-Score*) para verificar si los resultados cumplen con los criterios de éxito del negocio antes de pasar a producción. |
| **Deployment** | Integración del modelo predictivo en los sistemas operativos de NexoPlay para que la Directora de Retención y el equipo de Marketing puedan automatizar el envío de campañas, incentivos y correos de re-enganche dirigidos específicamente a los usuarios en riesgo. |