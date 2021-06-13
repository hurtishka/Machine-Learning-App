from flask_wtf import FlaskForm
from wtforms import SelectField, FileField, DecimalField, StringField
from wtforms.fields.core import FloatField
from wtforms.fields.simple import SubmitField
from wtforms.validators import DataRequired

class ClassificationForm(FlaskForm):
    
    input_file = FileField(
        label='Выберите файл',
        validators=[DataRequired('Выберите файл')]
    )

    classification_model = SelectField(
        label='Выберите модель'
    )

class LearningForm(FlaskForm):

    algorithms = SelectField(
        label='Выберите алгоритм'
    )

    max_depth = DecimalField(
        label='Введите максимальную глубину дерева',
    )

    criterion_clf = SelectField(
        label='Выберите критерий',
    )
    
    criterion_rgr = SelectField(
        label='Выберите критерий',
    )
    n_neighbors = DecimalField(
        label='Введите количество соседей',
    )

    algorithm_knn = SelectField(
        label='Выберите алгоритм для kNN',
    )
    kernel = SelectField(
        label='Выберите ядро для SVM',
    )
    
    c = DecimalField(
        label='Выберите атрибут C',
    )

    max_iter = DecimalField(
        label='Выберите максимальное количество итераций',
    )

    var_smoothing = FloatField(
        label='Выберите атрибут var_smoothing',
    )

    penalty = SelectField(
        label='Выберите штраф для LR',
    )

    solver = SelectField(
        label='Выберите решающий алгоритм для LR',
    )

    label_name = StringField(
        label='Введите название ключевого признака',
    )
    
    input_file = FileField(
        label='Выберите набор данных',
    )
    
    choose_algo = SubmitField(label=('Вывести параметры'))
    submit = SubmitField(label=('Обучить модель'))

    