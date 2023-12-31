# Generated by Django 4.2.5 on 2023-09-27 03:09

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='algo',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='pics',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('frame', models.ImageField(upload_to='algoviz/images/')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='algoviz.algo')),
            ],
        ),
    ]
