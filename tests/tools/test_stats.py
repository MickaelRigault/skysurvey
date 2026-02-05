import numpy as np 
from skysurvey.tools.stats import skewed_gaussian_pdf

# tests for the func skewed_gaussian
def test_skewed_gaussian_xx_str():
    xx_input="-5:5:0.001"
    loc=0.0
    xx, pdf = skewed_gaussian_pdf(xx=xx_input, loc=loc, scale_low=0.5, scale_high=1.5)

    assert isinstance(xx, np.ndarray)
    assert isinstance(pdf, np.ndarray)
    assert pdf.shape == xx.shape
    assert np.all(pdf >= 0)
    assert pdf[xx < loc].mean() < pdf[xx > loc].mean()

def test_skewed_gaussian_pdf_xx_array():
    xx_input = np.linspace(-5, 5,  1000)
    loc=0.0
    xx, pdf = skewed_gaussian_pdf(xx=xx_input, loc=loc, scale_low=0.5, scale_high=1.5)

    assert xx is xx_input
    assert len(xx) > 0 
    assert isinstance(xx, np.ndarray) 
    assert isinstance(pdf, np.ndarray)
    assert pdf.shape == xx.shape
    assert np.all(pdf >= 0)
    assert pdf[xx < loc].mean() < pdf[xx > loc].mean()

def test_skewed_gaussian_pdf_symmetric_case():
    scale = 1.0
    xx_input=np.array([-1,1])
    xx, pdf = skewed_gaussian_pdf(xx=xx_input, loc=0.0, scale_low=scale, scale_high=scale)

    assert np.isclose(pdf[0], pdf[1]) 