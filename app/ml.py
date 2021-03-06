"""Machine learning functions"""

import logging
import random
import pickle

from fastapi import APIRouter
import pandas as pd
from pydantic import BaseModel, Field, validator

log = logging.getLogger(__name__)
router = APIRouter()


class Item(BaseModel):
    """Use this data model to parse the request body JSON."""

    x1: float = Field(..., example=3.14)
    x2: int = Field(..., example=-42)
    x3: str = Field(..., example='banjo')

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame([dict(self)])

    @validator('x1')
    def x1_must_be_positive(cls, value):
        """Validate that x1 is a positive number."""
        assert value > 0, f'x1 == {value}, must be > 0'
        return value


class House(BaseModel):
    """Use this data model to parse the request body JSON."""
    bedrooms: int
    total_rooms: float
    house_age: float

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame([dict(self)])


@router.post('/predict/{description}')
async def predict(description: str):
    """
    Make random baseline predictions for classification problem 🔮

    ### Request Body
    - `x1`: positive float
    - `x2`: integer
    - `x3`: string

    ### Response
    - `prediction`: boolean, at random
    - `predict_proba`: float between 0.5 and 1.0, 
    representing the predicted class's probability

    Replace the placeholder docstring and fake predictions with your own model.
    """
    # return "ok"
    filename = 'notebooks/finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    print(type(loaded_model))
    print(loaded_model)
    # description = "Our space is a mix of a hostel and a home."
    y = [description]
    x = ['description'] # Series
    df_new = pd.DataFrame(y, columns= x)
    df_new['description']
    # return "ok"
    print("df")
    print(df_new['description'])
    pred = loaded_model.predict(df_new['description'])
    print(pred)
    # X_new = item.to_df()
    # log.info(X_new)
    # y_pred = random.choice([True, False])
    # y_pred_proba = random.random() / 2 + 0.5
    return {
        'description': description,
        'prediction_price': pred[0]
    }
