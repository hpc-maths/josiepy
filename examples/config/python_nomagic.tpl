{% extends 'python.tpl'%}
{% block input %}
  {{ super() | comment_magics }}
{% endblock input %}
