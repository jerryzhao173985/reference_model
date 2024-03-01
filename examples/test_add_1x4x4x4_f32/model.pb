node {
  name: "placeholder_0"
  op: "Placeholder"
  attr {
    key: "_user_specified_name"
    value {
      s: "placeholder_0"
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 4
        }
        dim {
          size: 4
        }
        dim {
          size: 1
        }
      }
    }
  }
}
node {
  name: "placeholder_1"
  op: "Placeholder"
  attr {
    key: "_user_specified_name"
    value {
      s: "placeholder_1"
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 4
        }
        dim {
          size: 4
        }
        dim {
          size: 4
        }
      }
    }
  }
}
node {
  name: "result"
  op: "AddV2"
  input: "placeholder_0"
  input: "placeholder_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Identity"
  op: "Identity"
  input: "result"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
versions {
  producer: 1581
}
