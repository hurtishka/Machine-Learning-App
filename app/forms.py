from flask_wtf import FlaskForm
from wtforms import SelectField, FileField, DecimalField
from wtforms.validators import DataRequired

class ClassificationForm(FlaskForm):
    
    input_file = FileField(
        label='Выберите файл',
        validators=[DataRequired('Выберите файл')]
    )

    classification_model = SelectField(
        label='Выберите модель'
    )
