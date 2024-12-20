# TensorBoard: Руководство по использованию

---

## Введение
TensorBoard — это мощный инструмент для визуализации и мониторинга процесса обучения моделей машинного обучения. Он позволяет отслеживать метрики, визуализировать графики, изображения и даже гистограммы. В этом руководстве мы рассмотрим, как использовать TensorBoard для отслеживания прогресса обучения модели, с акцентом на метрику `g/total`. Этот инструмент особенно полезен для моделей генеративных состязательных сетей (GAN), но также может быть использован для других типов моделей.

---

## Как использовать TensorBoard

### 1. Запуск TensorBoard
1. **Установка TensorBoard**:
   Если TensorBoard ещё не установлен, выполните следующую команду в терминале:
   ```bash
   pip install tensorboard
   ```

2. **Запуск TensorBoard**:
   Откройте TensorBoard, указав путь к папке с логами. Для этого выполните команду:
   ```bash
   tensorboard --logdir=replace/with/your/logs/folder/path
   ```
   - Замените `replace/with/your/logs/folder/path` на фактический путь к вашим логам.
   - Если вы используете Applio локально, вы можете запустить TensorBoard, запустив файл `run-tensorboard.bat`.

3. **Доступ к TensorBoard**:
   После запуска TensorBoard откройте веб-браузер и перейдите по адресу `http://localhost:6006`. Если порт 6006 занят, TensorBoard автоматически выберет другой порт, и адрес будет указан в терминале.

---

### 2. Настройка TensorBoard
1. **Переход на вкладку "Scalars"**:
   После открытия TensorBoard перейдите на вкладку "Scalars" (Скаляры). Здесь вы увидите графики различных метрик, включая `g/total`.

2. **Настройка сглаживания**:
   - Найдите метрику `g/total` в верхней части страницы.
   - Установите сглаживание (smoothing) на `0.950` или `0.987` для лучшего отображения графика. Сглаживание помогает уменьшить шум в графиках.

3. **Автоматическая перезагрузка данных**:
   - Нажмите на значок шестеренки (⚙️) в правом верхнем углу страницы.
   - Включите опцию автоматической перезагрузки данных каждые 30 секунд, чтобы видеть обновления в реальном времени.

4. **Работа с графиками**:
   - Под каждым графиком есть три кнопки:
     - Первая — для увеличения размера графика.
     - Вторая — для отключения оси Y.
     - Третья — для подгонки данных под график.
   - Снимите галочку с опции **"ignore outliers in chart scaling"**. Это помогает избежать искажения графика из-за выбросов.

---

### 3. Низшая точка
Низшая точка — это момент, когда график достигает такой низкой точки, которая больше не повторяется. Во время обучения могут быть несколько таких точек, которые нужно протестировать, чтобы найти оптимальную модель и предотвратить её переобучение.

- **Как найти низшую точку**:
  - Обратите внимание на количество шагов (steps) в этой точке.
  - Найдите эпохи (epochs) с этим или ближайшим шагом в сохранённых точках.
  - Сохраните модель в этой точке и протестируйте её на валидационном датасете.

---

## Мониторинг других метрик

### Основные метрики
- **`loss/g/total`**: Общая потеря генератора. Это ключевая метрика, которую нужно отслеживать. Она должна уменьшаться в процессе обучения.
- **`loss/d/total`**: Общая потеря дискриминатора. В начале обучения это значение может увеличиваться, но в дальнейшем должно стабилизироваться или уменьшаться. Если оно продолжает расти, это может указывать на то, что дискриминатор становится слишком сильным.
- **`loss/g/mel`**: Показывает точность модели в воспроизведении мел-спектрограммы из вашего датасета. Если это значение не уменьшается, это может указывать на проблемы с генерацией аудио.
- **`loss/g/kl`**: Отражает расхождение Кульбака-Лейблера (KL divergence). Это значение должно уменьшаться, чтобы генератор создавал данные с похожим распределением латентных переменных.

### Дополнительные метрики
- **`learning_rate`**: Текущая скорость обучения. Это значение может изменяться в зависимости от вашей стратегии обучения.
- **`grad_norm_d` и `grad_norm_g`**: Нормы градиентов для дискриминатора и генератора. Эти метрики помогают отслеживать, нет ли проблем с градиентами (например, взрывных или исчезающих градиентов).
- **`loss/g/fm`**: Потеря сопоставления признаков. Эта метрика поощряет генератор создавать данные с похожими промежуточными представлениями.

### Изображения
- **`slice/mel_org`**: Визуализация мел-спектрограммы целевого аудиосегмента.
- **`slice/mel_gen`**: Визуализация мел-спектрограммы сгенерированного аудиосегмента.
- **`all/mel`**: Визуализация мел-спектрограммы всего целевого аудио.

Сравнивая эти изображения, можно визуально оценить качество сгенерированного аудио.

---

## Термины потерь
- **Discriminator Loss (loss_disc)**: Показывает, насколько хорошо дискриминатор различает реальные и сгенерированные данные.
- **Generator Loss (loss_gen)**: Показывает, насколько хорошо генератор обманывает дискриминатор.
- **Feature Matching Loss (loss_fm)**: Поощряет генератор создавать данные с похожими промежуточными представлениями.
- **Mel Spectrogram Loss (loss_mel)**: Сравнивает мел-спектрограммы реальных и сгенерированных данных.
- **KL Divergence Loss (loss_kl)**: Поощряет генератор создавать данные с похожим распределением латентных переменных.

---

## Интерпретация данных
- **Общий тренд**: Ищите уменьшение общей потери генератора (`loss/g/total`) и стабильность или увеличение потери дискриминатора (`loss/d/total`). Это помогает понять, идет ли обучение в правильном направлении.
- **Норма градиента**: Отслеживайте нормы градиентов, чтобы избежать их чрезмерного увеличения или уменьшения. Это важно для стабильности обучения.
- **Компоненты потерь**: Анализируйте отдельные компоненты потерь, чтобы понять, как работают различные аспекты модели.
- **Мел-спектрограммы**: Сравните изображения мел-спектрограмм, чтобы визуально оценить качество сгенерированного аудио.

---
