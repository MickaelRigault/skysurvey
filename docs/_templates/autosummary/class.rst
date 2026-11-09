{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :inherited-members:
   :show-inheritance:

   {% block methods %}
   {%- set public_methods = methods | reject("equalto", "__init__") | list %}
   {%- if public_methods %}
   .. rubric:: Methods

   .. autosummary::
   {% for item in public_methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {%- endif %}
   {% endblock %}

   {% block attributes %}
   {%- if attributes %}
   .. rubric:: Attributes and properties

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {%- endif %}
   {% endblock %}
